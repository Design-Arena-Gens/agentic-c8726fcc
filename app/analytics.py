"""Performance analytics and evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from .model import FailureClassifier, compute_classification_metrics


@dataclass
class EvaluationResult:
    roc_points: pd.DataFrame
    pr_points: pd.DataFrame
    confusion: pd.DataFrame
    summary: Dict[str, float]


def evaluate_model(model: FailureClassifier, df: pd.DataFrame) -> EvaluationResult:
    if not model.fitted:
        raise RuntimeError("Model must be trained before evaluation.")

    probabilities = model.predict_proba(df)
    metrics = compute_classification_metrics(df["failure_event"].to_numpy(), probabilities)
    predictions = (probabilities >= 0.5).astype(int)

    roc_df = pd.DataFrame({"fpr": metrics["fprs"], "tpr": metrics["tprs"]})
    pr_df = pd.DataFrame({"precision": metrics["precisions"], "recall": metrics["recalls"]})

    labels = df["failure_event"].to_numpy()
    tp = np.logical_and(predictions == 1, labels == 1).sum()
    fp = np.logical_and(predictions == 1, labels == 0).sum()
    fn = np.logical_and(predictions == 0, labels == 1).sum()
    tn = np.logical_and(predictions == 0, labels == 0).sum()
    confusion_df = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Actual 0", "Actual 1"],
        columns=["Pred 0", "Pred 1"],
    )

    summary = {
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "f1": metrics["f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "threshold_optimal": metrics["threshold_optimal"],
    }

    return EvaluationResult(roc_points=roc_df, pr_points=pr_df, confusion=confusion_df, summary=summary)


def summarize_feature_importance(model: FailureClassifier) -> pd.DataFrame:
    """Return feature importance as a tidy dataframe."""

    if not model.fitted:
        raise RuntimeError("Model must be trained before requesting importance.")

    importance = model.feature_importance()
    if importance.empty:
        return pd.DataFrame(columns=["feature", "importance"])

    return importance.reset_index().rename(columns={"index": "feature", 0: "importance"})


def run_performance_study(
    df: pd.DataFrame,
    base_model: FailureClassifier,
) -> pd.DataFrame:
    probabilities = base_model.predict_proba(df)
    metrics = compute_classification_metrics(df["failure_event"].to_numpy(), probabilities)
    curve = pd.DataFrame(
        {
            "fpr": metrics["fprs"],
            "tpr": metrics["tprs"],
            "threshold": metrics["thresholds"],
            "precision": metrics["precisions"],
            "recall": metrics["recalls"],
        }
    )
    curve["miss_rate"] = 1 - curve["tpr"]
    curve["cost_score"] = curve["miss_rate"] * 0.7 + curve["fpr"] * 0.3
    optimal = curve.sort_values("cost_score").head(10)
    return optimal.reset_index(drop=True)
