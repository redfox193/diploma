from __future__ import annotations

import bisect
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import EnvironmentConfig, LineData
from .models import Analytics, Break, DeadheadTrip, Lunch, Route, Shift, Trip


@dataclass
class _DepartureEvent:
    """Единичная точка принятия решения -- рейс из расписания."""
    line_id: str
    departure_cp: str
    arrival_cp: str
    departure_time: float
    travel_time: float


@dataclass
class _BusState:
    """Внутреннее состояние водителя (автобуса)."""
    bus_id: str
    home_cp: str
    current_cp: Optional[str]
    shift_duration: float
    work_duration_before_lunch: float
    lunch_duration: float
    used: bool = False
    available_time: float = 0.0
    last_service_arrival: float = -math.inf
    busy_reason: Optional[str] = None
    destination_cp: Optional[str] = None
    last_line_id: Optional[str] = None
    shift_start_time: Optional[float] = None
    lunch_taken: bool = False
    lunch_start_time: Optional[float] = None


@dataclass
class _CandidateBus:
    """Кандидат-водитель для назначения на рейс."""
    bus: _BusState
    requires_deadhead: bool
    deadhead_time: float
    rest_time: float
    priority_reward: float
    source_cp: Optional[str]


class Environment:
    """Симулятор для оффлайн-обучения RL-агента задачи CSP."""

    _FEATURES_PER_CP = 5
    _FEATURES_PER_LINE = 3
    _FEATURES_PER_CANDIDATE = 6

    def __init__(self, config: EnvironmentConfig) -> None:
        self._config = config

        self._lines: Dict[str, LineData] = {
            line.id: line for line in config.schedule_data.lines
        }
        self._cp_ids: List[str] = sorted(
            cp.id for cp in config.schedule_data.control_points
        )
        self._cp_index: Dict[str, int] = {
            cp_id: idx for idx, cp_id in enumerate(self._cp_ids)
        }
        self._num_cps: int = len(self._cp_ids)

        self._departure_sequence = self._build_departure_sequence()
        self._total_departures = len(self._departure_sequence)
        if self._total_departures == 0:
            raise ValueError("Расписание не содержит ни одного рейса.")

        self._cp_departure_times: Dict[str, List[float]] = defaultdict(list)
        for dep in self._departure_sequence:
            self._cp_departure_times[dep.departure_cp].append(dep.departure_time)
        for times in self._cp_departure_times.values():
            times.sort()

        self._target_action_size = config.target_bus_set_size
        self._short_horizon = config.control_points.short_term_horizon
        self._min_rest = config.shifts.min_rest_duration
        self._time_norm = 60.0

        self._fleet_distribution = self._build_fleet_distribution()

        self._buses: Dict[str, _BusState] = {}
        self._current_step: int = 0
        self._current_departure: Optional[_DepartureEvent] = None
        self._current_time: float = 0.0
        self._current_candidates: List[Optional[_CandidateBus]] = []
        self._current_action_mask: np.ndarray = np.zeros(
            self._target_action_size, dtype=np.float32
        )
        self._total_deadhead_time: float = 0.0
        self._accumulated_reward: float = 0.0
        self._final_reward_value: float = 0.0
        self._invalid_actions: int = 0
        self._uncovered_departures: int = 0
        self._history: List[dict] = []
        self._next_auto_bus_idx: int = 0

    def reset(self) -> np.ndarray:
        """Сброс окружения в начальное состояние. Возвращает начальное наблюдение."""
        self._buses = {}
        for bus_id, cp_id in self._fleet_distribution:
            self._buses[bus_id] = _BusState(
                bus_id=bus_id,
                home_cp=cp_id,
                current_cp=cp_id,
                shift_duration=self._config.shifts.shift_duration,
                work_duration_before_lunch=self._config.shifts.work_duration_before_lunch,
                lunch_duration=self._config.shifts.lunch_duration,
            )
        self._next_auto_bus_idx = len(self._buses)
        self._current_step = 0
        self._current_departure = self._departure_sequence[0]
        self._current_time = self._current_departure.departure_time
        self._total_deadhead_time = 0.0
        self._accumulated_reward = 0.0
        self._final_reward_value = 0.0
        self._invalid_actions = 0
        self._uncovered_departures = 0
        self._history = []
        self._current_candidates = []
        self._current_action_mask = np.zeros(
            self._target_action_size, dtype=np.float32
        )
        return self._build_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Выполнить один шаг: назначить водителя на текущий рейс."""
        if self._current_departure is None:
            raise RuntimeError("Необходимо вызвать reset() перед step().")

        if (
            action < 0
            or action >= self._target_action_size
            or self._current_action_mask[action] == 0
        ):
            self._invalid_actions += 1
            action = self._fallback_action_index()

        candidate = self._current_candidates[action]
        if candidate is None:
            candidate = self._auto_assign_candidate()

        reward = self._apply_action(candidate)
        done = False

        self._current_step += 1
        if self._current_step >= self._total_departures:
            done = True
            final_r = self._final_reward()
            reward += final_r
            self._final_reward_value = final_r
            next_state = self._blank_state()
            self._current_departure = None
        else:
            self._current_departure = self._departure_sequence[self._current_step]
            self._current_time = self._current_departure.departure_time
            next_state = self._build_observation()

        self._accumulated_reward += reward
        info = self._build_info(candidate)
        return next_state, reward, done, info

    def state_dim(self) -> int:
        """Размерность вектора состояния."""
        return int(self._blank_state().shape[0])

    def action_dim(self) -> int:
        """Размерность пространства действий."""
        return self._target_action_size

    def get_episode_size(self) -> int:
        """Число шагов в одном эпизоде (== число рейсов в расписании)."""
        return self._total_departures

    @property
    def action_mask(self) -> np.ndarray:
        return self._current_action_mask.copy()

    def get_shifts(self) -> List[Shift]:
        """Построить список водительских смен из истории взаимодействия."""
        bus_events: Dict[str, List[dict]] = defaultdict(list)
        for record in self._history:
            bus_events[record["bus_id"]].append(record)

        shifts: List[Shift] = []
        for bus_id, events in bus_events.items():
            events.sort(key=lambda e: e["start_time"])
            shift_events = []
            for ev in events:
                if ev["event"] == "trip":
                    shift_events.append(Trip(
                        line_id=ev["line_id"],
                        start_time=ev["start_time"],
                        end_time=ev["end_time"],
                    ))
                elif ev["event"] == "lunch":
                    shift_events.append(Lunch(
                        cp_id=ev["cp_id"],
                        start_time=ev["start_time"],
                        end_time=ev["end_time"],
                    ))
                elif ev["event"] == "deadhead":
                    shift_events.append(DeadheadTrip(
                        cp_id_from=ev["cp_id_from"],
                        cp_id_to=ev["cp_id_to"],
                        start_time=ev["start_time"],
                        end_time=ev["end_time"],
                    ))
                elif ev["event"] == "break":
                    shift_events.append(Break(
                        cp_id=ev["cp_id"],
                        start_time=ev["start_time"],
                        end_time=ev["end_time"],
                    ))

            if not shift_events:
                continue

            departure_time = shift_events[0].start_time
            shifts.append(Shift(
                bus_id=bus_id,
                departure_time=departure_time,
                events=shift_events,
            ))

        shifts.sort(key=lambda s: s.departure_time)
        return shifts

    def get_routes_info(self) -> List[Route]:
        """Возвращает список маршрутов из конфигурации."""
        return [
            Route(name=r.name, line_id_A=r.line_id_A, line_id_B=r.line_id_B)
            for r in self._config.schedule_data.routes
        ]

    def get_analytics(self) -> Analytics:
        """Статистика по завершённому эпизоду."""
        total_bus_used = sum(1 for bus in self._buses.values() if bus.used)
        return Analytics(
            total_deadhead_time=self._total_deadhead_time,
            total_bus_used=total_bus_used,
            accumulated_reward=self._accumulated_reward,
            final_reward=self._final_reward_value,
        )

    def _build_departure_sequence(self) -> List[_DepartureEvent]:
        """Построить хронологическую последовательность рейсов из расписания."""
        sequence: List[_DepartureEvent] = []
        for entry in self._config.schedule_data.schedule:
            for dep in entry.departures:
                line = self._lines.get(dep.line_id)
                if line is None:
                    raise KeyError(
                        f"Неизвестный line_id '{dep.line_id}' в КП '{entry.cp_id}'"
                    )
                sequence.append(_DepartureEvent(
                    line_id=line.id,
                    departure_cp=line.cp_id_from,
                    arrival_cp=line.cp_id_to,
                    departure_time=dep.time,
                    travel_time=line.travel_time_min,
                ))
        sequence.sort(key=lambda d: d.departure_time)
        return sequence

    def _build_fleet_distribution(self) -> List[Tuple[str, str]]:
        """Равномерно распределить водителей по контрольным остановкам."""
        fleet: List[Tuple[str, str]] = []
        per_cp = self._config.bus_fleet_size // self._num_cps
        remainder = self._config.bus_fleet_size % self._num_cps
        bus_idx = 0
        for i, cp_id in enumerate(self._cp_ids):
            count = per_cp + (1 if i < remainder else 0)
            for _ in range(count):
                fleet.append((f"bus-{bus_idx}", cp_id))
                bus_idx += 1
        return fleet

    def _blank_state(self) -> np.ndarray:
        cp_size = self._num_cps * self._FEATURES_PER_CP
        line_size = self._FEATURES_PER_LINE
        bus_size = self._target_action_size * self._FEATURES_PER_CANDIDATE
        return np.zeros(cp_size + line_size + bus_size, dtype=np.float32)

    def _build_observation(self) -> np.ndarray:
        self._sync_bus_positions()
        cp_features = self._encode_control_points()
        line_features = self._encode_current_line()
        candidates = self._build_action_candidates()
        self._current_candidates = candidates
        self._current_action_mask = np.array(
            [1.0 if c is not None else 0.0 for c in candidates],
            dtype=np.float32,
        )
        bus_features = self._encode_candidates(candidates)
        return np.concatenate([cp_features, line_features, bus_features]).astype(
            np.float32
        )

    def _sync_bus_positions(self) -> None:
        """Обновить позиции водителей, завершивших рейс/перегон до текущего момента."""
        for bus in self._buses.values():
            if bus.busy_reason and self._current_time >= bus.available_time:
                bus.current_cp = bus.destination_cp
                if bus.busy_reason == "trip":
                    bus.last_service_arrival = bus.available_time
                bus.busy_reason = None
                bus.destination_cp = None
            if bus.busy_reason is None:
                self._maybe_trigger_lunch(bus)

    def _encode_control_points(self) -> np.ndarray:
        features: List[float] = []
        now = self._current_time
        for cp_id in self._cp_ids:
            n_lt = self._count_future_departures(cp_id, now, math.inf)
            n_st = self._count_future_departures(cp_id, now, self._short_horizon)
            n_a, n_o = self._count_buses_at_cp(cp_id)
            features.extend([
                self._normalize_cp_index(cp_id),
                n_lt,
                n_st,
                n_a,
                n_o,
            ])
        return np.array(features, dtype=np.float32)

    def _encode_current_line(self) -> np.ndarray:
        dep = self._current_departure
        assert dep is not None
        return np.array([
            self._normalize_cp_index(dep.departure_cp),
            self._normalize_cp_index(dep.arrival_cp),
            dep.travel_time / self._time_norm,
        ], dtype=np.float32)

    def _encode_candidates(
        self, candidates: Sequence[Optional[_CandidateBus]]
    ) -> np.ndarray:
        features: List[float] = []
        for cand in candidates:
            if cand is None:
                features.extend([0.0] * self._FEATURES_PER_CANDIDATE)
                continue
            bus = cand.bus
            cp_idx = (
                self._normalize_cp_index(bus.current_cp)
                if bus.current_cp is not None
                else 0.0
            )
            remaining_shift = self._normalize_time(self._remaining_shift_time(bus))
            time_until_lunch = self._normalize_time(self._time_until_lunch(bus))
            features.extend([
                1.0 if bus.used else 0.0,
                self._normalize_time(cand.rest_time),
                cp_idx,
                self._normalize_time(cand.deadhead_time),
                remaining_shift,
                time_until_lunch,
            ])
        return np.array(features, dtype=np.float32)

    def _build_action_candidates(self) -> List[Optional[_CandidateBus]]:
        dep = self._current_departure
        assert dep is not None

        available: List[Tuple[_BusState, bool, float, float]] = []
        used_buses: List[Tuple[_BusState, bool, float, float]] = []
        travel_time = dep.travel_time

        for bus in self._buses.values():
            self._maybe_trigger_lunch(bus)
            if bus.busy_reason is not None:
                continue
            if bus.available_time > self._current_time:
                continue

            rest = self._rest_time(bus)

            if bus.current_cp == dep.departure_cp and rest >= self._min_rest:
                if not self._shift_can_finish(bus, 0.0, travel_time):
                    continue
                available.append((bus, False, 0.0, rest))
                if bus.used:
                    used_buses.append((bus, False, 0.0, rest))
            else:
                deadhead = self._get_deadhead_time(bus.current_cp, dep.departure_cp)
                if math.isfinite(deadhead) and rest > deadhead + self._min_rest:
                    if not self._shift_can_finish(bus, deadhead, travel_time):
                        continue
                    available.append((bus, True, deadhead, rest))
                    if bus.used:
                        used_buses.append((bus, True, deadhead, rest))

        def _make_candidate(
            b: _BusState, req: bool, dh: float, r: float
        ) -> _CandidateBus:
            return _CandidateBus(
                bus=b,
                requires_deadhead=req,
                deadhead_time=dh,
                rest_time=r,
                priority_reward=0.0,
                source_cp=b.current_cp,
            )

        used_no_dh = sorted(
            [_make_candidate(b, req, dh, r) for b, req, dh, r in available
             if b.used and not req],
            key=lambda c: c.rest_time, reverse=True,
        )
        used_with_dh = sorted(
            [_make_candidate(b, req, dh, r) for b, req, dh, r in available
             if b.used and req],
            key=lambda c: c.deadhead_time,
        )
        unused_no_dh = [
            _make_candidate(b, req, dh, r) for b, req, dh, r in available
            if not b.used and not req
        ]
        unused_with_dh = [
            _make_candidate(b, req, dh, r) for b, req, dh, r in available
            if not b.used and req
        ]

        ordered_groups = [used_no_dh, used_with_dh, unused_no_dh, unused_with_dh]

        candidates: List[Optional[_CandidateBus]] = []
        for group in ordered_groups:
            for cand in group:
                candidates.append(cand)
                if len(candidates) == self._target_action_size:
                    break
            if len(candidates) == self._target_action_size:
                break

        while len(candidates) < self._target_action_size:
            candidates.append(None)

        self._assign_priority_rewards(candidates, used_buses)
        return candidates

    def _assign_priority_rewards(
        self,
        candidates: Sequence[Optional[_CandidateBus]],
        used_buses: List[Tuple[_BusState, bool, float, float]],
    ) -> None:
        n_used = len(used_buses)
        if n_used == 0:
            return
        used_buses.sort(key=lambda item: item[3], reverse=True)
        ranking: Dict[str, int] = {
            entry[0].bus_id: rank for rank, entry in enumerate(used_buses)
        }
        for cand in candidates:
            if cand is None or not cand.bus.used:
                continue
            rank = ranking.get(cand.bus.bus_id, n_used - 1)
            cand.priority_reward = (n_used - rank) / n_used

    def _apply_action(self, candidate: _CandidateBus) -> float:
        dep = self._current_departure
        assert dep is not None
        bus = candidate.bus
        was_used = bus.used
        bus.used = True

        if bus.shift_start_time is None:
            bus.shift_start_time = max(
                0.0, self._current_time - candidate.deadhead_time
            )

        if candidate.requires_deadhead and candidate.source_cp is not None:
            dh_start = self._current_time - candidate.deadhead_time
            self._history.append({
                "event": "deadhead",
                "bus_id": bus.bus_id,
                "cp_id_from": candidate.source_cp,
                "cp_id_to": dep.departure_cp,
                "start_time": dh_start,
                "end_time": self._current_time,
            })

        bus.last_line_id = dep.line_id
        bus.busy_reason = "trip"
        bus.destination_cp = dep.arrival_cp
        arrival_time = dep.departure_time + dep.travel_time
        bus.available_time = arrival_time
        bus.current_cp = None

        self._total_deadhead_time += candidate.deadhead_time

        self._history.append({
            "event": "trip",
            "bus_id": bus.bus_id,
            "line_id": dep.line_id,
            "start_time": dep.departure_time,
            "end_time": arrival_time,
        })

        unused_penalty = 0.0 if was_used else 1.0
        deadhead_penalty = candidate.deadhead_time
        rest_reward = candidate.priority_reward if was_used else 0.0
        demand_penalty = 0.0
        if candidate.requires_deadhead and candidate.source_cp is not None:
            demand_penalty = self._demand_penalty(candidate.source_cp, dep.arrival_cp)

        rw = self._config.reward
        return (
            -rw.step_unused_penalty * unused_penalty
            - rw.step_deadhead_penalty * deadhead_penalty
            + rw.step_rest_reward * rest_reward
            - rw.step_demand_penalty * demand_penalty
        )

    def _final_reward(self) -> float:
        rw = self._config.reward
        buses_used = sum(1 for bus in self._buses.values() if bus.used)
        return (
            -rw.final_buses * buses_used
            - rw.final_deadhead * self._total_deadhead_time
        )

    def _build_info(self, candidate: _CandidateBus) -> dict:
        buses_used = sum(1 for bus in self._buses.values() if bus.used)
        return {
            "current_time": self._current_time,
            "bus_id": candidate.bus.bus_id,
            "requires_deadhead": float(candidate.requires_deadhead),
            "deadhead_time": candidate.deadhead_time,
            "buses_used": float(buses_used),
            "total_deadhead_time": self._total_deadhead_time,
            "invalid_actions": float(self._invalid_actions),
            "uncovered_departures": float(self._uncovered_departures),
        }

    def _fallback_action_index(self) -> int:
        for idx, mask_val in enumerate(self._current_action_mask):
            if mask_val == 1:
                return idx
        self._uncovered_departures += 1
        self._current_candidates[0] = self._auto_assign_candidate()
        self._current_action_mask[0] = 1
        return 0

    def _auto_assign_candidate(self) -> _CandidateBus:
        if self._current_departure is None:
            raise RuntimeError("Нет активного рейса для автоназначения.")
        bus = self._spawn_bus(self._current_departure.departure_cp)
        return _CandidateBus(
            bus=bus,
            requires_deadhead=False,
            deadhead_time=0.0,
            rest_time=float("inf"),
            priority_reward=0.0,
            source_cp=bus.current_cp,
        )

    def _spawn_bus(self, cp_id: str) -> _BusState:
        if len(self._buses) >= self._config.bus_fleet_size:
            raise RuntimeError(
                "Превышена вместимость парка. Увеличьте bus_fleet_size."
            )
        bus_id = f"auto-{self._next_auto_bus_idx}"
        self._next_auto_bus_idx += 1
        bus = _BusState(
            bus_id=bus_id,
            home_cp=cp_id,
            current_cp=cp_id,
            shift_duration=self._config.shifts.shift_duration,
            work_duration_before_lunch=self._config.shifts.work_duration_before_lunch,
            lunch_duration=self._config.shifts.lunch_duration,
        )
        self._buses[bus_id] = bus
        return bus

    def _shift_can_finish(
        self, bus: _BusState, deadhead_time: float, trip_time: float
    ) -> bool:
        upcoming = deadhead_time + trip_time
        if bus.shift_start_time is None:
            return upcoming <= bus.shift_duration
        elapsed = self._current_time - bus.shift_start_time
        if elapsed >= bus.shift_duration:
            return False
        return (elapsed + upcoming) <= bus.shift_duration

    def _remaining_shift_time(self, bus: _BusState) -> float:
        if bus.shift_start_time is None:
            return bus.shift_duration
        elapsed = self._current_time - bus.shift_start_time
        return max(0.0, bus.shift_duration - elapsed)

    def _time_until_lunch(self, bus: _BusState) -> float:
        """Время до обязательного обеда. 0 если обед уже был."""
        if bus.lunch_taken:
            return 0.0
        if bus.shift_start_time is None:
            return bus.work_duration_before_lunch
        elapsed = self._current_time - bus.shift_start_time
        return max(0.0, bus.work_duration_before_lunch - elapsed)

    def _needs_lunch(self, bus: _BusState) -> bool:
        if bus.lunch_taken or bus.shift_start_time is None:
            return False
        elapsed = self._current_time - bus.shift_start_time
        return elapsed >= bus.work_duration_before_lunch

    def _maybe_trigger_lunch(self, bus: _BusState) -> None:
        if self._needs_lunch(bus) and bus.busy_reason is None:
            self._start_lunch(bus)

    def _start_lunch(self, bus: _BusState) -> None:
        bus.busy_reason = "lunch"
        bus.lunch_taken = True
        bus.lunch_start_time = self._current_time
        if bus.shift_start_time is None:
            bus.shift_start_time = self._current_time
        bus.destination_cp = bus.current_cp
        bus.available_time = self._current_time + bus.lunch_duration
        self._history.append({
            "event": "lunch",
            "bus_id": bus.bus_id,
            "cp_id": bus.current_cp,
            "start_time": self._current_time,
            "end_time": self._current_time + bus.lunch_duration,
        })

    def _normalize_cp_index(self, cp_id: Optional[str]) -> float:
        if cp_id is None:
            return 0.0
        return (self._cp_index[cp_id] + 1) / (self._num_cps + 1)

    def _normalize_time(self, minutes: float) -> float:
        if minutes in (math.inf, -math.inf):
            return 1.0
        return minutes / self._time_norm

    def _count_future_departures(
        self, cp_id: str, start_time: float, horizon: float
    ) -> int:
        times = self._cp_departure_times.get(cp_id, [])
        start_idx = bisect.bisect_left(times, start_time)
        if horizon == math.inf:
            return len(times) - start_idx
        end_time = start_time + horizon
        end_idx = bisect.bisect_right(times, end_time)
        return max(0, end_idx - start_idx)

    def _count_buses_at_cp(self, cp_id: str) -> Tuple[int, int]:
        total = 0
        used = 0
        for bus in self._buses.values():
            if bus.busy_reason is not None:
                continue
            if bus.current_cp == cp_id and bus.available_time <= self._current_time:
                total += 1
                if bus.used:
                    used += 1
        return total, used

    def _rest_time(self, bus: _BusState) -> float:
        if not bus.used:
            return math.inf
        return max(0.0, self._current_time - bus.last_service_arrival)

    def _get_deadhead_time(self, origin: Optional[str], destination: str) -> float:
        if origin is None:
            origin = destination
        return self._config.deadhead_time(origin, destination)

    def _demand_penalty(self, source_cp: str, target_cp: str) -> float:
        demand_source = self._cp_demand_score(source_cp)
        demand_target = self._cp_demand_score(target_cp)
        return 1.0 if demand_source > demand_target else 0.0

    def _cp_demand_score(self, cp_id: str) -> float:
        n_s = self._count_future_departures(
            cp_id, self._current_time, self._short_horizon
        )
        _, n_o = self._count_buses_at_cp(cp_id)
        return n_s / (n_o + 1.0)
