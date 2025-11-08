from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from app.analytics import evaluate_model, run_performance_study, summarize_feature_importance
from app.data import EnvironmentConfig, generate_baseline_dataset, generate_streaming_sample
from app.model import FEATURE_COLUMNS, load_or_train_model
from app.simulation import TurbineConfig, apply_protection_logic, scenario_sweep, simulate_turbine_response


st.set_page_config(
    page_title="Wind Turbine Failure Protection Lab",
    layout="wide",
    page_icon="🌬️",
)


@st.cache_data(show_spinner=False)
def cached_dataset(samples: int, env_dict: Dict[str, Any]) -> pd.DataFrame:
    env = EnvironmentConfig(**env_dict)
    return generate_baseline_dataset(n_samples=samples, env=env)


def environment_controls() -> EnvironmentConfig:
    st.sidebar.header("Environment Controls")
    mean_wind = st.sidebar.slider("Mean wind speed (m/s)", 6.0, 18.0, 11.5)
    wind_std = st.sidebar.slider("Wind variability σ (m/s)", 1.0, 6.0, 3.5)
    ambient_temp = st.sidebar.slider("Ambient temperature (°C)", -10.0, 35.0, 12.0)
    humidity = st.sidebar.slider("Humidity (%)", 10.0, 95.0, 55.0)
    turbulence = st.sidebar.slider("Turbulence intensity", 0.02, 0.5, 0.1)
    seed = st.sidebar.number_input("Random seed", 0, 9999, 42, step=1)

    return EnvironmentConfig(
        mean_wind_speed=mean_wind,
        wind_speed_std=wind_std,
        ambient_temperature=ambient_temp,
        humidity=humidity,
        turbulence_intensity=turbulence,
        seed=int(seed),
    )


def turbine_config_controls() -> TurbineConfig:
    st.sidebar.header("Protection Settings")
    rated_speed = st.sidebar.slider("Rated rotor speed (rpm)", 10.0, 20.0, 16.5)
    critical_speed = st.sidebar.slider("Critical rotor speed (rpm)", rated_speed + 0.5, 24.0, 20.5)
    vibration_trip = st.sidebar.slider("Vibration trip level (g)", 1.0, 4.0, 2.6)
    gearbox_limit = st.sidebar.slider("Gearbox temperature limit (°C)", 80.0, 130.0, 108.0)
    generator_limit = st.sidebar.slider("Generator temperature limit (°C)", 70.0, 130.0, 100.0)
    yaw_limit = st.sidebar.slider("Yaw misalignment limit (°)", 5.0, 30.0, 18.0)
    braking_gain = st.sidebar.slider("Braking gain", 0.2, 1.5, 0.75)
    pitch_gain = st.sidebar.slider("Pitch gain", 0.2, 1.2, 0.6)

    return TurbineConfig(
        rated_rotor_speed=rated_speed,
        critical_rotor_speed=critical_speed,
        vibration_trip_g=vibration_trip,
        gearbox_temp_limit=gearbox_limit,
        generator_temp_limit=generator_limit,
        yaw_error_limit=yaw_limit,
        braking_gain=braking_gain,
        pitch_gain=pitch_gain,
    )


def render_dataset_summary(df: pd.DataFrame) -> None:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Observations", f"{len(df):,}")
    with c2:
        st.metric("Failure rate", f"{df['failure_event'].mean():.2%}")
    with c3:
        st.metric("Avg power output (MW)", f"{df['power_output_mw'].mean():.2f}")
    with c4:
        st.metric("Protection trips", int(df['protection_triggered'].sum()))

    fig = px.scatter(
        df.sample(min(1200, len(df)), random_state=0),
        x="wind_speed",
        y="power_output_mw",
        color="failure_event",
        size="vibration_level",
        labels={"failure_event": "Failure"},
        title="Power Curve with Failure Events",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_failure_dashboard(model, df: pd.DataFrame) -> None:
    evaluation = evaluate_model(model, df)
    c1, c2 = st.columns(2)
    with c1:
        roc_fig = px.area(
            evaluation.roc_points,
            x="fpr",
            y="tpr",
            title=f"ROC Curve (AUC={evaluation.summary['roc_auc']:.3f})",
        )
        roc_fig.add_shape(
            type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1, name="baseline"
        )
        st.plotly_chart(roc_fig, use_container_width=True)
    with c2:
        pr_fig = px.area(
            evaluation.pr_points,
            x="recall",
            y="precision",
            title=f"Precision-Recall Curve (AUC={evaluation.summary['pr_auc']:.3f})",
        )
        st.plotly_chart(pr_fig, use_container_width=True)

    st.subheader("Confusion Matrix")
    st.dataframe(evaluation.confusion.style.background_gradient(cmap="Blues"), use_container_width=True)

    st.subheader("Summary Metrics")
    metrics = evaluation.summary
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
    m2.metric("PR AUC", f"{metrics['pr_auc']:.3f}")
    m3.metric("F1 score", f"{metrics['f1']:.3f}")
    m4.metric("Optimal threshold", f"{metrics['threshold_optimal']:.2f}")

    st.subheader("Feature Importance")
    importance_df = summarize_feature_importance(model)
    if not importance_df.empty:
        fig = px.bar(
            importance_df,
            x="importance",
            y="feature",
            orientation="h",
            title="Feature Importance",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("The current model does not expose feature importance metrics.")


def render_control_simulation(df: pd.DataFrame, cfg: TurbineConfig) -> None:
    st.subheader("Protection System Simulation")
    evaluated = apply_protection_logic(df.sample(min(len(df), 500), random_state=1), cfg)
    status_counts = evaluated["operating_status"].value_counts(normalize=True)
    st.metric("Mean curtailed power (MW)", f"{evaluated['curtailed_power_mw'].mean():.2f}")
    st.metric("Expected power loss (MW)", f"{evaluated['expected_power_loss'].sum():.2f}")

    fig = px.histogram(
        evaluated,
        x="expected_power_loss",
        nbins=30,
        title="Distribution of Power Loss Per Event",
    )
    st.plotly_chart(fig, use_container_width=True)

    status_fig = px.bar(
        status_counts,
        title="Operating Status Mix",
        labels={"index": "Status", "value": "Ratio"},
    )
    st.plotly_chart(status_fig, use_container_width=True)

    st.subheader("Scenario Sweeps")
    sweep = scenario_sweep(
        evaluated,
        cfg,
        pitch_offsets=np.linspace(0.3, 0.9, 5),
        braking_gains=np.linspace(0.5, 1.2, 6),
    )
    sweep_fig = px.scatter(
        sweep,
        x="power_loss_mw",
        y="failure_rate",
        color="pitch_gain",
        size="normal_operation_ratio",
        hover_data=["braking_gain"],
        title="Protection Tuning Trade-offs",
    )
    st.plotly_chart(sweep_fig, use_container_width=True)
    st.dataframe(sweep, use_container_width=True)


def render_streaming_monitor(df: pd.DataFrame, model, cfg: TurbineConfig) -> None:
    st.subheader("Live Stream Monitoring")
    sample = df.sample(1, random_state=None).iloc[0]
    streaming_df = generate_streaming_sample(sample)
    streaming_df["predicted_failure_prob"] = model.predict_proba(streaming_df)

    fig = px.line(
        streaming_df,
        y="predicted_failure_prob",
        title="Predicted Failure Probability Over Time",
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)

    st.write("Recent telemetry snapshot")
    st.dataframe(streaming_df.tail(10), use_container_width=True)

    response = simulate_turbine_response(sample, cfg)
    st.json(response)


def main() -> None:
    st.title("Wind Turbine Failure Protection & Performance Lab")
    st.caption(
        "Simulate turbine behaviour, evaluate protection logic, and analyse ML-driven failure prediction."
    )

    env_cfg = environment_controls()
    turbine_cfg = turbine_config_controls()

    with st.spinner("Generating synthetic dataset…"):
        dataset = cached_dataset(samples=6000, env_dict=env_cfg.__dict__)

    retrain = st.sidebar.checkbox("Force retrain model", value=False)
    with st.spinner("Training or loading machine learning model…"):
        model = load_or_train_model(dataset, force_retrain=retrain)

    tabs = st.tabs(
        [
            "Dataset",
            "Failure Prediction",
            "Protection Simulation",
            "Streaming Monitor",
            "Performance Study",
        ]
    )

    with tabs[0]:
        render_dataset_summary(dataset)

    with tabs[1]:
        render_failure_dashboard(model, dataset)

    with tabs[2]:
        render_control_simulation(dataset, turbine_cfg)

    with tabs[3]:
        render_streaming_monitor(dataset, model, turbine_cfg)

    with tabs[4]:
        st.subheader("Threshold Optimisation Study")
        study = run_performance_study(dataset, model)
        st.dataframe(study, use_container_width=True)
        chart = px.scatter(
            study,
            x="fpr",
            y="miss_rate",
            color="threshold",
            size="cost_score",
            title="False Positive vs Miss Rate Trade-off",
        )
        st.plotly_chart(chart, use_container_width=True)

    st.sidebar.download_button(
        label="Download dataset (CSV)",
        data=dataset.to_csv(index=False).encode("utf-8"),
        file_name="wind_turbine_dataset.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
