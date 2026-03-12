#!/usr/bin/env python3
import math
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from typing import Optional
from logging_utils import get_logger

logger = get_logger(__name__)


class AnomalyDetector:

    def __init__(self, z_threshold: float = 3.0, contamination: float = 0.05):
        self.z_threshold = z_threshold
        self.contamination = contamination  # expected fraction of anomalies

    def zscore_flag(
        self,
        values: pd.Series,
        mean: float,
        std: float
    ) -> pd.Series:
        """
        Flag values more than z_threshold standard deviations from the
        established baseline mean. Returns a Series of z-scores.
        """
        try:
            if not isinstance(values, pd.Series):
                error_msg = f"Expected pd.Series, got {type(values)}"
                logger.error(error_msg)
                raise TypeError(error_msg)
            
            if std == 0 or not math.isfinite(std):
                logger.warning(f"Zero or non-finite std ({std}), returning zero z-scores")
                return pd.Series([0.0] * len(values))
            
            if not math.isfinite(mean):
                error_msg = f"Non-finite mean value: {mean}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            z_scores = (values - mean).abs() / std
            logger.debug(f"Computed z-scores: min={z_scores.min():.4f}, max={z_scores.max():.4f}")
            return z_scores
        except Exception as e:
            error_msg = f"Error computing z-scores: {e}"
            logger.error(error_msg, exc_info=True)
            print(f"ERROR: {error_msg}")
            raise

    def isolation_forest_flag(self, df: pd.DataFrame, numeric_cols: list[str]) -> np.ndarray:
        """
        Multivariate anomaly detection across all numeric channels simultaneously.
        IsolationForest returns -1 for anomalies, 1 for normal points.
        Scores closer to -1 indicate stronger anomalies.
        """
        try:
            if df.empty:
                error_msg = "DataFrame is empty"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Check that all required columns exist
            missing_cols = [col for col in numeric_cols if col not in df.columns]
            if missing_cols:
                error_msg = f"Missing required columns: {missing_cols}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            model = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            )
            
            try:
                X = df[numeric_cols].fillna(df[numeric_cols].median())
                if X.empty or X.shape[0] == 0:
                    error_msg = "No valid data after preprocessing"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            except Exception as e:
                error_msg = f"Error preprocessing data for IsolationForest: {e}"
                logger.error(error_msg, exc_info=True)
                raise
            
            try:
                model.fit(X)
                labels = model.predict(X)          # -1 = anomaly, 1 = normal
                scores = model.decision_function(X)  # lower = more anomalous
                logger.debug(f"IsolationForest: {np.sum(labels == -1)} anomalies detected out of {len(labels)} points")
            except Exception as e:
                error_msg = f"Error running IsolationForest: {e}"
                logger.error(error_msg, exc_info=True)
                raise

            return labels, scores
        except Exception as e:
            error_msg = f"Critical error in isolation_forest_flag: {e}"
            logger.error(error_msg, exc_info=True)
            print(f"ERROR: {error_msg}")
            raise

    def run(
        self,
        df: pd.DataFrame,
        numeric_cols: list[str],
        baseline: dict,
        method: str = "both"
    ) -> pd.DataFrame:
        try:
            if df.empty:
                error_msg = "Cannot run detection on empty DataFrame"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            if not numeric_cols:
                error_msg = "No numeric columns specified"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            result = df.copy()

            # --- Z-score per channel ---
            if method in ("zscore", "both"):
                try:
                    for col in numeric_cols:
                        if col not in df.columns:
                            logger.warning(f"Column {col} not in DataFrame, skipping z-score")
                            result[f"{col}_zscore"] = None
                            result[f"{col}_zscore_flag"] = None
                            continue
                        
                        stats = baseline.get(col)
                        if stats and isinstance(stats, dict) and stats.get("count", 0) >= 30:
                            try:
                                z_scores = self.zscore_flag(df[col], stats["mean"], stats["std"])
                                result[f"{col}_zscore"] = z_scores.round(4)
                                result[f"{col}_zscore_flag"] = z_scores > self.z_threshold
                            except Exception as e:
                                logger.error(f"Error computing z-score for {col}: {e}", exc_info=True)
                                result[f"{col}_zscore"] = None
                                result[f"{col}_zscore_flag"] = None
                        else:
                            # Not enough baseline history yet — flag as unknown
                            result[f"{col}_zscore"] = None
                            result[f"{col}_zscore_flag"] = None
                except Exception as e:
                    error_msg = f"Error in z-score computation: {e}"
                    logger.error(error_msg, exc_info=True)
                    raise

            # --- IsolationForest across all channels ---
            if method in ("isolation", "both"):
                try:
                    labels, scores = self.isolation_forest_flag(df, numeric_cols)
                    result["if_label"] = labels          # -1 or 1
                    result["if_score"] = scores.round(4) # continuous anomaly score
                    result["if_flag"] = labels == -1
                except Exception as e:
                    error_msg = f"Error in IsolationForest: {e}"
                    logger.error(error_msg, exc_info=True)
                    # If isolation fails but zscore succeeded, continue with zscore only
                    if method == "isolation":
                        raise
                    logger.warning("IsolationForest failed, continuing with z-score only")
                    result["if_label"] = None
                    result["if_score"] = None
                    result["if_flag"] = False

            # --- Consensus flag: anomalous by at least one method ---
            if method == "both":
                try:
                    zscore_flags = [
                        result[f"{col}_zscore_flag"]
                        for col in numeric_cols
                        if f"{col}_zscore_flag" in result.columns
                        and result[f"{col}_zscore_flag"].notna().any()
                    ]
                    if zscore_flags:
                        any_zscore = pd.concat(zscore_flags, axis=1).any(axis=1)
                        if_flag = result.get("if_flag", pd.Series([False] * len(result)))
                        result["anomaly"] = any_zscore | if_flag
                    else:
                        result["anomaly"] = result.get("if_flag", pd.Series([False] * len(result)))
                except Exception as e:
                    error_msg = f"Error computing consensus anomaly flag: {e}"
                    logger.error(error_msg, exc_info=True)
                    # Fallback to IsolationForest flag if available
                    result["anomaly"] = result.get("if_flag", pd.Series([False] * len(result)))

            logger.info(f"Detection completed: method={method}, rows={len(result)}")
            return result
        except Exception as e:
            error_msg = f"Critical error in detector.run: {e}"
            logger.error(error_msg, exc_info=True)
            print(f"ERROR: {error_msg}")
            raise
