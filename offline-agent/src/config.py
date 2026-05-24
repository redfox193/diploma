from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import yaml


@dataclass
class ShiftsConfig:
    shift_duration: int
    work_duration_before_lunch: int
    lunch_duration: int
    min_rest_duration: int


@dataclass
class ControlPointsConfig:
    short_term_horizon: int


@dataclass
class RewardConfig:
    final_buses: float
    final_deadhead: float
    step_unused_penalty: float
    step_deadhead_penalty: float
    step_rest_reward: float
    step_demand_penalty: float


@dataclass
class ControlPointData:
    id: str
    name: str


@dataclass
class LineData:
    id: str
    name: str
    cp_id_from: str
    cp_id_to: str
    travel_time_min: float


@dataclass
class DepartureData:
    line_id: str
    time: float


@dataclass
class ScheduleEntry:
    cp_id: str
    departures: List[DepartureData]


@dataclass
class RouteData:
    name: str
    line_id_A: str
    line_id_B: str


@dataclass
class ScheduleData:
    control_points: List[ControlPointData]
    deadhead_matrix: Dict[str, Dict[str, float]]
    lines: List[LineData]
    schedule: List[ScheduleEntry]
    routes: List[RouteData] = field(default_factory=list)


@dataclass
class EnvironmentConfig:
    shifts: ShiftsConfig
    control_points: ControlPointsConfig
    target_bus_set_size: int
    reward: RewardConfig
    bus_fleet_size: int
    schedule_data_path: str
    schedule_data: ScheduleData = field(repr=False, default=None)  # type: ignore[assignment]

    def deadhead_time(self, origin: str, destination: str) -> float:
        if origin == destination:
            return 0.0
        return self.schedule_data.deadhead_matrix.get(origin, {}).get(
            destination, float("inf")
        )


def load_environment_config(config_path: str) -> EnvironmentConfig:
    config_path_obj = Path(config_path)

    with config_path_obj.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    shifts = ShiftsConfig(
        shift_duration=int(raw["shifts"]["shift_duration"]),
        work_duration_before_lunch=int(raw["shifts"]["work_duration_before_lunch"]),
        lunch_duration=int(raw["shifts"]["lunch_duration"]),
        min_rest_duration=int(raw["shifts"]["min_rest_duration"]),
    )

    control_points_cfg = ControlPointsConfig(
        short_term_horizon=int(raw["control_points"]["short_term_horizon"]),
    )

    reward = RewardConfig(
        final_buses=float(raw["reward"]["final_buses"]),
        final_deadhead=float(raw["reward"]["final_deadhead"]),
        step_unused_penalty=float(raw["reward"]["step_unused_penalty"]),
        step_deadhead_penalty=float(raw["reward"]["step_deadhead_penalty"]),
        step_rest_reward=float(raw["reward"]["step_rest_reward"]),
        step_demand_penalty=float(raw["reward"]["step_demand_penalty"]),
    )

    schedule_data_path = raw["schedule_data_path"]

    # Resolve schedule JSON path relative to the config file location
    schedule_json_path = config_path_obj.parent / schedule_data_path
    with schedule_json_path.open("r", encoding="utf-8") as f:
        schedule_raw = json.load(f)

    schedule_data = ScheduleData(
        control_points=[
            ControlPointData(id=cp["id"], name=cp["name"])
            for cp in schedule_raw["control_points"]
        ],
        deadhead_matrix={
            origin: {dest: float(t) for dest, t in mapping.items()}
            for origin, mapping in schedule_raw["deadhead_matrix"].items()
        },
        lines=[
            LineData(
                id=line["id"],
                name=line["name"],
                cp_id_from=line["cp_id_from"],
                cp_id_to=line["cp_id_to"],
                travel_time_min=float(line["travel_time_min"]),
            )
            for line in schedule_raw["lines"]
        ],
        schedule=[
            ScheduleEntry(
                cp_id=entry["cp_id"],
                departures=[
                    DepartureData(line_id=dep["line_id"], time=float(dep["time"]))
                    for dep in entry["departures"]
                ],
            )
            for entry in schedule_raw["schedule"]
        ],
        routes=[
            RouteData(
                name=route["name"],
                line_id_A=route["line_id_A"],
                line_id_B=route["line_id_B"],
            )
            for route in schedule_raw.get("routes", [])
        ],
    )

    return EnvironmentConfig(
        shifts=shifts,
        control_points=control_points_cfg,
        target_bus_set_size=int(raw["target_bus_set_size"]),
        reward=reward,
        bus_fleet_size=int(raw["bus_fleet_size"]),
        schedule_data_path=schedule_data_path,
        schedule_data=schedule_data,
    )


@dataclass
class PPOAgentConfig:
    hidden_layers: List[int]
    activation: str
    learning_rate: float
    gamma: float
    clip_epsilon: float
    gae_lambda: float
    epochs: int
    mini_batch_size: int
    rollout_steps: int
    entropy_coef: float
    value_coef: float
    max_grad_norm: float


def load_ppo_agent_config(config_path: str) -> PPOAgentConfig:
    with Path(config_path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    arch = raw["architecture"]
    ppo = raw["ppo"]

    return PPOAgentConfig(
        hidden_layers=arch["hidden_layers"],
        activation=arch["activation"],
        learning_rate=float(ppo["learning_rate"]),
        gamma=float(ppo["gamma"]),
        clip_epsilon=float(ppo["clip_epsilon"]),
        gae_lambda=float(ppo["gae_lambda"]),
        epochs=int(ppo["epochs"]),
        mini_batch_size=int(ppo["mini_batch_size"]),
        rollout_steps=int(ppo["rollout_steps"]),
        entropy_coef=float(ppo["entropy_coef"]),
        value_coef=float(ppo["value_coef"]),
        max_grad_norm=float(ppo["max_grad_norm"]),
    )


@dataclass
class PPOAgentTrainConfig:
    episodes: int
    display_reward_training_plot: bool
    display_policy_changing_plot: bool
    checkpoint_folder_name: str


def load_ppo_agent_train_config(config_path: str) -> PPOAgentTrainConfig:
    with Path(config_path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return PPOAgentTrainConfig(
        episodes=int(raw["episodes"]),
        display_reward_training_plot=bool(raw["display_reward_training_plot"]),
        display_policy_changing_plot=bool(raw["display_policy_changing_plot"]),
        checkpoint_folder_name=raw["checkpoint_folder_name"],
    )


@dataclass
class PPOAgentValidateConfig:
    number_of_runs: int


def load_ppo_agent_validate_config(config_path: str) -> PPOAgentValidateConfig:
    with Path(config_path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return PPOAgentValidateConfig(
        number_of_runs=int(raw["number_of_runs"]),
    )
