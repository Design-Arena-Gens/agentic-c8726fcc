"""Synthetic data generation utilities for the wind turbine simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class EnvironmentConfig:
    """Parameterization for the synthetic environment."""

    mean_wind_speed: float = 11.5
    wind_speed_std: float = 3.5
    ambient_temperature: float = 12.0
    humidity: float = 55.0
    turbulence_intensity: float = 0.1
    seed: Optional[int] = 42


def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)


def generate_baseline_dataset(
    n_samples: int = 6000,
    env: Optional[EnvironmentConfig] = None,
) -> pd.DataFrame:
    """Build a synthetic dataset capturing nominal and failure states."""

    env = env or EnvironmentConfig()
    rng = _rng(env.seed)

    wind_speed = rng.normal(env.mean_wind_speed, env.wind_speed_std, n_samples).clip(1, 25)
    turbulence = rng.gamma(2.0, env.turbulence_intensity, n_samples).clip(0.01, 0.5)
    rotor_speed = (
        (wind_speed * rng.uniform(8.5, 9.4, n_samples))
        * (1 - turbulence * rng.uniform(0.18, 0.35, n_samples))
    )
    rotor_speed = rotor_speed.clip(5.0, 22.0)

    pitch_command = rng.normal(6.5, 3.0, n_samples) - (wind_speed - env.mean_wind_speed) * 0.3
    pitch_command = pitch_command.clip(-2.0, 25.0)
    pitch_actual = pitch_command + rng.normal(0, 0.8, n_samples)

    nacelle_yaw_error = rng.normal(0, 4.0, n_samples) + turbulence * 15 * rng.uniform(
        -1, 1, n_samples
    )
    nacelle_yaw_error = nacelle_yaw_error.clip(-25, 25)

    ambient_temp = rng.normal(env.ambient_temperature, 2.5, n_samples)
    gearbox_temp = ambient_temp + rng.normal(40, 12, n_samples)
    generator_temp = ambient_temp + rng.normal(35, 10, n_samples)
    bearing_temp = ambient_temp + rng.normal(28, 8, n_samples)

    vibration_level = rng.normal(0.75, 0.2, n_samples)
    vibration_level += (turbulence - 0.1) * 1.8
    vibration_level = vibration_level.clip(0.2, 3.5)

    grid_demand = rng.normal(2.2, 0.8, n_samples).clip(0.5, 4.5)

    theoretical_cp = (1.0 - np.exp(-0.08 * wind_speed)) * np.cos(np.radians(pitch_actual))
    theoretical_cp = theoretical_cp.clip(0.1, 0.55)
    power_output = 0.5 * 1.225 * (np.pi * (60**2)) * (wind_speed**3) * theoretical_cp / 1e6
    power_output = power_output * rng.normal(0.92, 0.07, n_samples)
    power_output = power_output.clip(0, 5.5)

    reactive_power = power_output * rng.normal(0.32, 0.08, n_samples)

    failure_risk = np.zeros(n_samples)
    failure_risk += (vibration_level > 1.8) * 0.4
    failure_risk += (gearbox_temp > 90) * 0.35
    failure_risk += (generator_temp > 85) * 0.25
    failure_risk += (nacelle_yaw_error > 12) * 0.2
    failure_risk += (rotor_speed > 18.5) * 0.3
    failure_risk += (pitch_actual > 18) * 0.15

    maintenance_override = rng.uniform(0, 1, n_samples) < 0.05
    failure_risk += maintenance_override * 0.3

    raw_failure_prob = np.clip(failure_risk + rng.normal(0, 0.08, n_samples), 0, 1)
    failure_event = raw_failure_prob > 0.55

    protection_triggered = (
        (rotor_speed > 20)
        | (pitch_actual > 22)
        | (gearbox_temp > 105)
        | (vibration_level > 2.5)
    )

    curtailed_power = power_output.copy()
    curtailed_power[protection_triggered] *= rng.uniform(0.1, 0.55, size=protection_triggered.sum())

    df = pd.DataFrame(
        {
            "wind_speed": wind_speed,
            "turbulence": turbulence,
            "rotor_speed": rotor_speed,
            "pitch_command": pitch_command,
            "pitch_actual": pitch_actual,
            "nacelle_yaw_error": nacelle_yaw_error,
            "ambient_temperature": ambient_temp,
            "gearbox_temperature": gearbox_temp,
            "generator_temperature": generator_temp,
            "bearing_temperature": bearing_temp,
            "vibration_level": vibration_level,
            "grid_demand_factor": grid_demand,
            "power_output_mw": power_output,
            "reactive_power_mvar": reactive_power,
            "curtailed_power_mw": curtailed_power,
            "maintenance_override": maintenance_override.astype(float),
            "failure_event": failure_event.astype(int),
            "raw_failure_probability": raw_failure_prob,
            "protection_triggered": protection_triggered.astype(int),
        }
    )

    df["power_loss_mw"] = df["power_output_mw"] - df["curtailed_power_mw"]
    df["efficiency"] = df["curtailed_power_mw"] / df["power_output_mw"].replace(0, np.nan)
    df["efficiency"] = df["efficiency"].fillna(1.0).clip(0, 1.1)
    df["smoothed_failure_probability"] = pd.Series(raw_failure_prob).rolling(12, min_periods=1).mean()

    return df


def generate_streaming_sample(
    base_row: Optional[pd.Series] = None,
    env: Optional[EnvironmentConfig] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Produce a streaming-like batch of recent observations."""

    env = env or EnvironmentConfig(seed=seed)
    rng = _rng(seed if seed is not None else env.seed)
    n = 128

    if base_row is None:
        base_df = generate_baseline_dataset(n_samples=2048, env=env)
        base_row = base_df.sample(1, random_state=seed).iloc[0]

    perturb = rng.normal(0, 0.05, (n, base_row.size))
    base_vals = base_row.to_numpy()
    samples = base_vals + perturb * base_vals

    streaming_df = pd.DataFrame([base_row] * n, columns=base_row.index).reset_index(drop=True)
    values = streaming_df.to_numpy(dtype=float)
    values += perturb * values
    streaming_df = pd.DataFrame(values, columns=streaming_df.columns)

    # Introduce temporal trend
    trend = np.linspace(-0.1, 0.15, n)
    streaming_df["gearbox_temperature"] += trend * 15
    streaming_df["generator_temperature"] += trend * 12
    streaming_df["vibration_level"] += trend * 1.2
    streaming_df["rotor_speed"] += trend * 2.0
    if "smoothed_failure_probability" in streaming_df.columns:
        streaming_df["smoothed_failure_probability"] = (
            streaming_df["smoothed_failure_probability"].clip(0, 1)
        )

    return streaming_df
