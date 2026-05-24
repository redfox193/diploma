"""Базовые тесты для окружения."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure the src package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    ControlPointData,
    ControlPointsConfig,
    DepartureData,
    EnvironmentConfig,
    LineData,
    RewardConfig,
    ScheduleData,
    ScheduleEntry,
    ShiftsConfig,
    load_environment_config,
)
from src.environment import Environment
from src.models import Analytics, Break, DeadheadTrip, Lunch, Shift, Trip


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

def _make_small_config() -> EnvironmentConfig:
    """Маленький конфиг с 2 остановками и 6 рейсами для тестов."""
    schedule_data = ScheduleData(
        control_points=[
            ControlPointData(id="A", name="Stop A"),
            ControlPointData(id="B", name="Stop B"),
        ],
        deadhead_matrix={
            "A": {"A": 0.0, "B": 10.0},
            "B": {"A": 10.0, "B": 0.0},
        },
        lines=[
            LineData(id="L1", name="Line 1", cp_id_from="A", cp_id_to="B", travel_time_min=20.0),
            LineData(id="L2", name="Line 2", cp_id_from="B", cp_id_to="A", travel_time_min=20.0),
        ],
        schedule=[
            ScheduleEntry(cp_id="A", departures=[
                DepartureData(line_id="L1", time=100.0),
                DepartureData(line_id="L1", time=130.0),
                DepartureData(line_id="L1", time=160.0),
            ]),
            ScheduleEntry(cp_id="B", departures=[
                DepartureData(line_id="L2", time=125.0),
                DepartureData(line_id="L2", time=155.0),
                DepartureData(line_id="L2", time=185.0),
            ]),
        ],
    )

    return EnvironmentConfig(
        shifts=ShiftsConfig(
            shift_duration=480,
            work_duration_before_lunch=240,
            lunch_duration=30,
            min_rest_duration=3,
        ),
        control_points=ControlPointsConfig(short_term_horizon=300),
        target_bus_set_size=4,
        reward=RewardConfig(
            final_buses=4.0,
            final_deadhead=0.1,
            step_unused_penalty=4.0,
            step_deadhead_penalty=0.1,
            step_rest_reward=2.0,
            step_demand_penalty=1.0,
        ),
        bus_fleet_size=6,
        schedule_data_path="",
        schedule_data=schedule_data,
    )


@pytest.fixture
def small_config():
    return _make_small_config()


@pytest.fixture
def env(small_config):
    e = Environment(small_config)
    e.reset()
    return e


# ------------------------------------------------------------------ #
# Tests
# ------------------------------------------------------------------ #

class TestEnvironmentCreation:
    def test_reset_returns_observation(self, env: Environment):
        obs = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32
        assert obs.shape == (env.state_dim(),)

    def test_state_dim(self, env: Environment):
        num_cps = 2
        expected = num_cps * 5 + 3 + env.action_dim() * 6
        assert env.state_dim() == expected

    def test_action_dim(self, env: Environment):
        assert env.action_dim() == 4

    def test_get_episode_size(self, env: Environment):
        assert env.get_episode_size() == 6


class TestEpisodeRun:
    def test_full_episode_with_random_actions(self, env: Environment):
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            mask = env.action_mask
            valid_actions = np.where(mask == 1.0)[0]
            if len(valid_actions) == 0:
                action = 0
            else:
                action = int(valid_actions[0])
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1

        assert done is True
        assert steps == env.get_episode_size()
        assert isinstance(total_reward, float)

    def test_step_before_reset_raises(self, small_config):
        e = Environment(small_config)
        with pytest.raises(RuntimeError):
            e.step(0)

    def test_invalid_action_handled(self, env: Environment):
        env.reset()
        obs, reward, done, info = env.step(999)
        assert isinstance(obs, np.ndarray)
        assert info["invalid_actions"] >= 1.0


class TestGetShifts:
    def test_shifts_nonempty_after_episode(self, env: Environment):
        env.reset()
        done = False
        while not done:
            mask = env.action_mask
            valid = np.where(mask == 1.0)[0]
            action = int(valid[0]) if len(valid) > 0 else 0
            _, _, done, _ = env.step(action)

        shifts = env.get_shifts()
        assert len(shifts) > 0
        for shift in shifts:
            assert isinstance(shift, Shift)
            assert len(shift.events) > 0
            assert isinstance(shift.bus_id, str)

    def test_shift_events_are_typed(self, env: Environment):
        env.reset()
        done = False
        while not done:
            mask = env.action_mask
            valid = np.where(mask == 1.0)[0]
            action = int(valid[0]) if len(valid) > 0 else 0
            _, _, done, _ = env.step(action)

        shifts = env.get_shifts()
        for shift in shifts:
            for event in shift.events:
                assert isinstance(event, (Trip, Break, DeadheadTrip, Lunch))


class TestGetAnalytics:
    def test_analytics_after_episode(self, env: Environment):
        env.reset()
        done = False
        while not done:
            mask = env.action_mask
            valid = np.where(mask == 1.0)[0]
            action = int(valid[0]) if len(valid) > 0 else 0
            _, _, done, _ = env.step(action)

        analytics = env.get_analytics()
        assert isinstance(analytics, Analytics)
        assert analytics.total_bus_used > 0
        assert isinstance(analytics.total_deadhead_time, float)
        assert isinstance(analytics.accumulated_reward, float)
        assert isinstance(analytics.final_reward, float)
        assert analytics.final_reward < 0.0


class TestLoadConfig:
    def test_load_from_yaml(self):
        config_path = (
            Path(__file__).resolve().parent.parent / "src" / "config" / "environment.yaml"
        )
        if not config_path.exists():
            pytest.skip("environment.yaml not found")
        cfg = load_environment_config(str(config_path))
        assert cfg.bus_fleet_size == 260
        assert cfg.target_bus_set_size == 8
        assert cfg.shifts.shift_duration == 480
        assert len(cfg.schedule_data.control_points) == 4
        assert len(cfg.schedule_data.lines) == 12

    def test_real_env_creation(self):
        config_path = (
            Path(__file__).resolve().parent.parent / "src" / "config" / "environment.yaml"
        )
        if not config_path.exists():
            pytest.skip("environment.yaml not found")
        cfg = load_environment_config(str(config_path))
        e = Environment(cfg)
        obs = e.reset()
        assert obs.shape == (e.state_dim(),)
        assert e.get_episode_size() == 974
