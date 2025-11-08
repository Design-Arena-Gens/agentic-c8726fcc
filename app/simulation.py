"""Simulation utilities for wind turbine control and protection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


@dataclass
class TurbineConfig:
    """Control limits for the turbine."""

    rated_power_mw: float = 5.0
    rated_rotor_speed: float = 16.5
    critical_rotor_speed: float = 20.5
    pitch_limit_deg: float = 24.0
    vibration_trip_g: float = 2.6
    gearbox_temp_limit: float = 108.0
    generator_temp_limit: float = 100.0
    yaw_error_limit: float = 18.0
    reactive_limit: float = 2.4
    braking_gain: float = 0.75
    pitch_gain: float = 0.6


def simulate_turbine_response(
    sample: pd.Series,
    cfg: TurbineConfig,
) -> Dict[str, float]:
    """Simulate how the supervisory control would react to a data sample."""

    response: Dict[str, float] = {}

    rotor_speed = float(sample["rotor_speed"])
    vibration = float(sample["vibration_level"])
    gearbox_temp = float(sample["gearbox_temperature"])
    generator_temp = float(sample["generator_temperature"])
    yaw_error = abs(float(sample["nacelle_yaw_error"]))
    reactive_power = float(sample["reactive_power_mvar"])
    power_output = float(sample["curtailed_power_mw"])

    braking = 0.0
    pitch_override = 0.0
    yaw_correction = 0.0
    status = "normal"

    if rotor_speed > cfg.critical_rotor_speed:
        braking = min(1.0, (rotor_speed - cfg.critical_rotor_speed) / 4.5) * cfg.braking_gain
        pitch_override = min(cfg.pitch_limit_deg, (rotor_speed - cfg.rated_rotor_speed) * 1.35)
        status = "emergency_brake"
    elif rotor_speed > cfg.rated_rotor_speed:
        pitch_override = (rotor_speed - cfg.rated_rotor_speed) * cfg.pitch_gain
        status = "pitch_control"

    if vibration > cfg.vibration_trip_g:
        braking = max(braking, (vibration - cfg.vibration_trip_g) * 0.6)
        status = "vibration_trip"

    if gearbox_temp > cfg.gearbox_temp_limit or generator_temp > cfg.generator_temp_limit:
        pitch_override = max(pitch_override, 3.5)
        braking = max(braking, 0.35)
        status = "thermal_derate"

    if yaw_error > cfg.yaw_error_limit:
        yaw_correction = np.sign(sample["nacelle_yaw_error"]) * -min(4.0, yaw_error - cfg.yaw_error_limit)
        status = "yaw_alignment"

    if reactive_power > cfg.reactive_limit:
        response["reactive_spillover"] = reactive_power - cfg.reactive_limit
        status = "grid_support"

    curtailed_power = max(0.0, power_output * (1 - braking * 0.9))
    expected_power_loss = power_output - curtailed_power

    response.update(
        {
            "braking_setpoint": float(np.clip(braking, 0, 1)),
            "pitch_override_deg": float(np.clip(pitch_override, 0, cfg.pitch_limit_deg)),
            "yaw_correction_deg": float(yaw_correction),
            "expected_power_loss": float(expected_power_loss),
            "curtailed_power_mw": float(curtailed_power),
            "operating_status": status,
        }
    )
    return response


def apply_protection_logic(
    df: pd.DataFrame,
    cfg: TurbineConfig,
) -> pd.DataFrame:
    """Apply the control logic to an entire dataset."""

    records = []
    for _, sample in df.iterrows():
        response = simulate_turbine_response(sample, cfg)
        records.append(response)

    response_df = pd.DataFrame(records)
    merged = pd.concat([df.reset_index(drop=True), response_df], axis=1)
    merged["total_power_loss"] = merged["expected_power_loss"].cumsum()
    return merged


def scenario_sweep(
    df: pd.DataFrame,
    cfg: TurbineConfig,
    pitch_offsets: Iterable[float],
    braking_gains: Iterable[float],
) -> pd.DataFrame:
    """Sweep multiple control settings to understand performance impacts."""

    results = []
    for pitch_gain in pitch_offsets:
        for braking in braking_gains:
            scenario_cfg = TurbineConfig(
                rated_power_mw=cfg.rated_power_mw,
                rated_rotor_speed=cfg.rated_rotor_speed,
                critical_rotor_speed=cfg.critical_rotor_speed,
                pitch_limit_deg=cfg.pitch_limit_deg,
                vibration_trip_g=cfg.vibration_trip_g,
                gearbox_temp_limit=cfg.gearbox_temp_limit,
                generator_temp_limit=cfg.generator_temp_limit,
                yaw_error_limit=cfg.yaw_error_limit,
                reactive_limit=cfg.reactive_limit,
                braking_gain=braking,
                pitch_gain=pitch_gain,
            )
            evaluated = apply_protection_logic(df, scenario_cfg)
            failure_rate = evaluated["failure_event"].mean()
            power_loss = evaluated["expected_power_loss"].sum()
            avg_status = evaluated["operating_status"].value_counts(normalize=True).to_dict()

            results.append(
                {
                    "pitch_gain": pitch_gain,
                    "braking_gain": braking,
                    "failure_rate": failure_rate,
                    "power_loss_mw": power_loss,
                    "status_mix": avg_status,
                }
            )

    summary = pd.DataFrame(results)
    summary["normal_operation_ratio"] = summary["status_mix"].apply(lambda d: d.get("normal", 0.0))
    summary["yaw_alignment_ratio"] = summary["status_mix"].apply(lambda d: d.get("yaw_alignment", 0.0))
    summary["thermal_derate_ratio"] = summary["status_mix"].apply(lambda d: d.get("thermal_derate", 0.0))
    summary = summary.drop(columns=["status_mix"])
    return summary
