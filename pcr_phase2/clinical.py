from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from pcr_phase2.utils import load_json, save_json


@dataclass
class ClinicalPreprocessor:
    numeric_cols: List[str]
    categorical_cols: List[str]
    numeric_median: Dict[str, float]
    numeric_mean: Dict[str, float]
    numeric_std: Dict[str, float]
    categorical_mode: Dict[str, float]
    categorical_mappings: Dict[str, Dict[str, float]]
    missing_counts: Dict[str, int]
    feature_columns: List[str]
    age_mean: float
    age_std: float
    fitted: bool = False

    def __init__(self, numeric_cols: List[str], categorical_cols: List[str]) -> None:
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)
        self.numeric_median = {}
        self.numeric_mean = {}
        self.numeric_std = {}
        self.categorical_mode = {}
        self.categorical_mappings = {}
        self.missing_counts = {}
        self.feature_columns = self.numeric_cols + self.categorical_cols
        self.age_mean = 0.0
        self.age_std = 1.0
        self.fitted = False

    def fit(self, df: pd.DataFrame) -> None:
        for col in self.numeric_cols:
            series = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(dtype=float)
            self.missing_counts[col] = int(series.isna().sum())
            median = float(series.median()) if len(series.dropna()) > 0 else 0.0
            filled = series.fillna(median)
            mean = float(filled.mean()) if len(filled) > 0 else 0.0
            std = float(filled.std(ddof=0)) if len(filled) > 0 else 1.0
            if std < 1e-6:
                std = 1.0
            self.numeric_median[col] = median
            self.numeric_mean[col] = mean
            self.numeric_std[col] = std
            if col == "age":
                self.age_mean = mean
                self.age_std = std

        for col in self.categorical_cols:
            series = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(dtype=float)
            self.missing_counts[col] = int(series.isna().sum())
            valid = series.dropna()
            if len(valid) == 0:
                mode = 0.0
            else:
                mode = float(valid.mode(dropna=True).iloc[0])
            self.categorical_mode[col] = mode
            unique_values = sorted({float(value) for value in valid.tolist()})
            if len(unique_values) == 0:
                unique_values = [mode]
            self.categorical_mappings[col] = {str(value): float(value) for value in unique_values}

        self.feature_columns = self.numeric_cols + self.categorical_cols
        self.fitted = True

    def _resolve_numeric_value(self, row: pd.Series, col: str) -> float:
        value = pd.to_numeric(pd.Series([row.get(col, np.nan)]), errors="coerce").iloc[0]
        if pd.isna(value):
            value = self.numeric_median[col]
        return float(value)

    def _resolve_categorical_value(self, row: pd.Series, col: str) -> float:
        value = pd.to_numeric(pd.Series([row.get(col, np.nan)]), errors="coerce").iloc[0]
        if pd.isna(value):
            value = 0.0 if col == "menopause" else self.categorical_mode[col]
        mapping = self.categorical_mappings.get(col, {})
        encoded = mapping.get(str(float(value)))
        if encoded is None:
            encoded = float(value)
        return float(encoded)

    def transform_row(self, row: pd.Series) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("ClinicalPreprocessor must be fitted before transform.")
        features: List[float] = []
        for col in self.numeric_cols:
            value = self._resolve_numeric_value(row, col)
            value = (float(value) - self.numeric_mean[col]) / self.numeric_std[col]
            features.append(float(value))
        for col in self.categorical_cols:
            features.append(self._resolve_categorical_value(row, col))
        return np.asarray(features, dtype=np.float32)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return np.stack([self.transform_row(row) for _, row in df.iterrows()], axis=0)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "clinical_feature_version": 2,
            "clinical_num_cols": self.numeric_cols,
            "clinical_cat_cols": self.categorical_cols,
            "clinical_dim": len(self.feature_columns),
            "numeric_cols": self.numeric_cols,
            "categorical_cols": self.categorical_cols,
            "numeric_median": self.numeric_median,
            "numeric_mean": self.numeric_mean,
            "numeric_std": self.numeric_std,
            "categorical_mode": self.categorical_mode,
            "categorical_mappings": self.categorical_mappings,
            "missing_counts": self.missing_counts,
            "age_mean": self.age_mean,
            "age_std": self.age_std,
            "feature_columns": self.feature_columns,
        }

    def save_json(self, path: str) -> None:
        save_json(path, self.state_dict())

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "ClinicalPreprocessor":
        numeric_cols = list(state.get("clinical_num_cols", state.get("numeric_cols", [])))
        categorical_cols = list(state.get("clinical_cat_cols", state.get("categorical_cols", [])))
        obj = cls(
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
        )
        obj.numeric_median = {k: float(v) for k, v in state.get("numeric_median", {}).items()}
        obj.numeric_mean = {k: float(v) for k, v in state.get("numeric_mean", {}).items()}
        obj.numeric_std = {k: float(v) for k, v in state.get("numeric_std", {}).items()}
        obj.categorical_mode = {k: float(v) for k, v in state.get("categorical_mode", {}).items()}
        obj.categorical_mappings = {
            k: {str(inner_key): float(inner_value) for inner_key, inner_value in mapping.items()}
            for k, mapping in state.get("categorical_mappings", {}).items()
        }
        obj.missing_counts = {k: int(v) for k, v in state.get("missing_counts", {}).items()}
        obj.feature_columns = list(state.get("feature_columns", obj.numeric_cols + obj.categorical_cols))
        obj.age_mean = float(state.get("age_mean", obj.numeric_mean.get("age", 0.0)))
        obj.age_std = float(state.get("age_std", obj.numeric_std.get("age", 1.0)))
        obj.fitted = True
        return obj

    @classmethod
    def load_json(cls, path: str) -> "ClinicalPreprocessor":
        return cls.from_state_dict(load_json(path))

