from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Union


@dataclass
class Trip:
    """Водитель находился в рейсе."""
    line_id: str
    start_time: float
    end_time: float


@dataclass
class Break:
    """Водитель находился на отдыхе."""
    cp_id: str
    start_time: float
    end_time: float


@dataclass
class DeadheadTrip:
    """Водитель совершал холостую поездку."""
    cp_id_from: str
    cp_id_to: str
    start_time: float
    end_time: float


@dataclass
class Lunch:
    """Водитель находился на обеде."""
    cp_id: str
    start_time: float
    end_time: float


ShiftEvent = Union[Trip, Break, DeadheadTrip, Lunch]


@dataclass
class Shift:
    bus_id: str
    departure_time: float
    events: List[ShiftEvent] = field(default_factory=list)


@dataclass
class Analytics:
    total_deadhead_time: float
    total_bus_used: int
    accumulated_reward: float
    final_reward: float


@dataclass
class Route:
    """Маршрут, объединяющий две линии (направления A и B)."""
    name: str
    line_id_A: str
    line_id_B: str


@dataclass
class ValidateResult:
    """Результат валидации агента по нескольким запускам."""
    mean_accumulated_reward: float
    mean_total_reward: float
    best_total_bus_used: int
    best_total_deadhead_time: float
    best_final_reward: float
    best_accumulated_reward: float
    best_shifts: List[Shift]
