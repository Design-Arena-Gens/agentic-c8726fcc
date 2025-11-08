"""Custom logistic-regression style model for failure prediction."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "wind_speed",
    "turbulence",
    "rotor_speed",
    "pitch_actual",
    "nacelle_yaw_error",
    "ambient_temperature",
    "gearbox_temperature",
    "generator_temperature",
    "bearing_temperature",
    "vibration_level",
    "grid_demand_factor",
    "power_output_mw",
    "reactive_power_mvar",
    "curtailed_power_mw",
    "maintenance_override",
    "smoothed_failure_probability",
]

TARGET_COLUMN = "failure_event"


@dataclass
class TrainingReport:
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float


class FailureClassifier:
    """Lightweight logistic regression with feature scaling."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iter: int = 8000,
        l2_penalty: float = 0.001,
        random_state: int = 13,
    ) -> None:
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.l2_penalty = l2_penalty
        self.random_state = random_state

        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None
        self.fitted: bool = False
        self.last_metrics: Dict[str, float] = {}

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
        if self.feature_mean is None or self.feature_std is None:
            self.feature_mean = X.mean(axis=0)
            self.feature_std = X.std(axis=0) + 1e-6
        X_norm = (X - self.feature_mean) / self.feature_std
        return X_norm

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, df: pd.DataFrame) -> TrainingReport:
        rng = np.random.default_rng(self.random_state)
        X = self._prepare_features(df)
        y = df[TARGET_COLUMN].to_numpy(dtype=float)

        n_features = X.shape[1]
        self.weights = rng.normal(0, 0.1, size=n_features)
        self.bias = 0.0

        for step in range(self.max_iter):
            linear = X @ self.weights + self.bias
            preds = self._sigmoid(linear)
            error = preds - y

            grad_w = X.T @ error / len(y) + self.l2_penalty * self.weights
            grad_b = error.mean()

            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b

            if step % 1000 == 0:
                loss = -np.mean(y * np.log(preds + 1e-8) + (1 - y) * np.log(1 - preds + 1e-8))
                loss += 0.5 * self.l2_penalty * np.sum(self.weights**2)
                self.last_metrics["loss"] = float(loss)

        self.fitted = True
        probabilities = self.predict_proba(df)
        metrics = compute_classification_metrics(df[TARGET_COLUMN].to_numpy(), probabilities)
        self.last_metrics.update(metrics)

        return TrainingReport(
            roc_auc=metrics["roc_auc"],
            pr_auc=metrics["pr_auc"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1=metrics["f1"],
        )

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted or self.weights is None or self.feature_mean is None:
            raise RuntimeError("Model must be fitted before predictions.")
        X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
        X_norm = (X - self.feature_mean) / self.feature_std
        return self._sigmoid(X_norm @ self.weights + self.bias)

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(df) >= threshold).astype(int)

    def feature_importance(self) -> pd.Series:
        if self.weights is None:
            raise RuntimeError("Model must be fitted before requesting importance.")
        return pd.Series(np.abs(self.weights), index=FEATURE_COLUMNS).sort_values(ascending=False)


def compute_classification_metrics(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    thresholds = np.linspace(0, 1, 101)
    tprs = []
    fprs = []
    precisions = []
    recalls = []
    for thr in thresholds:
        preds = (scores >= thr).astype(int)
        tp = np.logical_and(preds == 1, y_true == 1).sum()
        fp = np.logical_and(preds == 1, y_true == 0).sum()
        fn = np.logical_and(preds == 0, y_true == 1).sum()
        tn = np.logical_and(preds == 0, y_true == 0).sum()

        tpr = tp / (tp + fn + 1e-8)
        fpr = fp / (fp + tn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tpr

        tprs.append(tpr)
        fprs.append(fpr)
        precisions.append(precision)
        recalls.append(recall)

    fprs_arr = np.array(fprs)
    tprs_arr = np.array(tprs)
    precisions_arr = np.array(precisions)
    recalls_arr = np.array(recalls)

    order = np.argsort(fprs_arr)
    fprs_arr = fprs_arr[order]
    tprs_arr = tprs_arr[order]
    roc_auc = float(np.trapz(tprs_arr, fprs_arr))

    pr_order = np.argsort(recalls_arr)
    recalls_arr = recalls_arr[pr_order]
    precisions_arr = precisions_arr[pr_order]
    pr_auc = float(np.trapz(precisions_arr, recalls_arr))

    default_preds = (scores >= 0.5).astype(int)
    tp = np.logical_and(default_preds == 1, y_true == 1).sum()
    fp = np.logical_and(default_preds == 1, y_true == 0).sum()
    fn = np.logical_and(default_preds == 0, y_true == 1).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "threshold_optimal": float(thresholds[np.argmax(np.array(precisions) * np.array(recalls))]),
        "fprs": fprs_arr,
        "tprs": tprs_arr,
        "precisions": precisions_arr,
        "recalls": recalls_arr,
        "thresholds": thresholds,
    }


def model_cache_dir() -> Path:
    cache = Path(".model_cache")
    cache.mkdir(exist_ok=True)
    return cache


def cache_path(name: str = "failure_classifier.pkl") -> Path:
    return model_cache_dir() / name


def load_or_train_model(df: pd.DataFrame, force_retrain: bool = False) -> FailureClassifier:
    path = cache_path()
    if path.exists() and not force_retrain:
        with path.open("rb") as f:
            model: FailureClassifier = pickle.load(f)
            model.fitted = True
            return model

    model = FailureClassifier()
    model.fit(df)
    with path.open("wb") as f:
        pickle.dump(model, f)
    return model
