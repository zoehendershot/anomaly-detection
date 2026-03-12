#!/usr/bin/env python3
import json
import math
import boto3
from datetime import datetime
from typing import Optional
from logging_utils import get_logger, sync_log_file_to_s3

logger = get_logger(__name__)

s3 = boto3.client("s3")

class BaselineManager:
    """
    Maintains a per-channel running baseline using Welford's online algorithm,
    which computes mean and variance incrementally without storing all past data.
    """

    def __init__(self, bucket: str, baseline_key: str = "state/baseline.json"):
        self.bucket = bucket
        self.baseline_key = baseline_key

    def load(self) -> dict:
        try:
            response = s3.get_object(Bucket=self.bucket, Key=self.baseline_key)
            content = response["Body"].read()
            baseline = json.loads(content)
            logger.info(f"Loaded baseline from s3://{self.bucket}/{self.baseline_key}")
            return baseline
        except s3.exceptions.NoSuchKey:
            logger.info(f"Baseline file not found, returning empty baseline: s3://{self.bucket}/{self.baseline_key}")
            return {}
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse baseline JSON from s3://{self.bucket}/{self.baseline_key}: {e}"
            logger.error(error_msg, exc_info=True)
            print(f"ERROR: {error_msg}")
            raise
        except Exception as e:
            error_msg = f"Failed to load baseline from s3://{self.bucket}/{self.baseline_key}: {e}"
            logger.error(error_msg, exc_info=True)
            print(f"ERROR: {error_msg}")
            raise

    def save(self, baseline: dict):
        baseline["last_updated"] = datetime.utcnow().isoformat()
        baseline_json = json.dumps(baseline, indent=2)
        try:
            s3.put_object(
                Bucket=self.bucket,
                Key=self.baseline_key,
                Body=baseline_json,
                ContentType="application/json"
            )
            logger.info("EVENT: Baseline updated - saved to s3://%s/%s", self.bucket, self.baseline_key)

            s3.put_object(
                Bucket=self.bucket,
                Key="state/baseline-final.json",
                Body=baseline_json,
                ContentType="application/json"
            )
            logger.info("Saved final baseline copy to s3://%s/state/baseline-final.json", self.bucket)

            # Keep a fresh copy of the local app log in S3 whenever baseline.json is pushed.
            sync_log_file_to_s3(s3, self.bucket, s3_key="logs/anomaly-api.log", logger=logger)
        except Exception as e:
            error_msg = f"Failed to save baseline to S3: {e}"
            logger.error(error_msg, exc_info=True)
            print(f"ERROR: {error_msg}")
            raise


    def update(self, baseline: dict, channel: str, new_values: list[float]) -> dict:
        """
        Welford's online algorithm for numerically stable mean and variance.
        Each channel tracks: count, mean, M2 (sum of squared deviations).
        Variance = M2 / count, std = sqrt(variance).
        """
        try:
            if not isinstance(new_values, list) or len(new_values) == 0:
                logger.warning(f"Invalid new_values for channel {channel}: {new_values}")
                return baseline

            if channel not in baseline:
                baseline[channel] = {"count": 0, "mean": 0.0, "M2": 0.0}

            state = baseline[channel]

            for value in new_values:
                try:
                    float_value = float(value)
                    if not math.isfinite(float_value):
                        logger.warning(f"Skipping non-finite value for channel {channel}: {value}")
                        continue
                    state["count"] += 1
                    delta = float_value - state["mean"]
                    state["mean"] += delta / state["count"]
                    delta2 = float_value - state["mean"]
                    state["M2"] += delta * delta2
                except (ValueError, TypeError) as e:
                    logger.warning(f"Skipping invalid value for channel {channel}: {value} ({e})")
                    continue

            # Only compute std once we have enough observations
            if state["count"] >= 2:
                try:
                    variance = state["M2"] / state["count"]
                    state["std"] = math.sqrt(variance) if variance >= 0 else 0.0
                except Exception as e:
                    logger.error(f"Error computing std for channel {channel}: {e}", exc_info=True)
                    state["std"] = 0.0
            else:
                state["std"] = 0.0

            baseline[channel] = state
            logger.info(f"EVENT: Baseline updated for channel '{channel}': count={state['count']}, mean={state['mean']:.4f}, std={state['std']:.4f}")
            logger.debug(f"Updated baseline for channel {channel}: count={state['count']}, mean={state['mean']:.4f}, std={state['std']:.4f}")
            return baseline
        except Exception as e:
            error_msg = f"Failed to update baseline for channel {channel}: {e}"
            logger.error(error_msg, exc_info=True)
            print(f"ERROR: {error_msg}")
            raise

    def get_stats(self, baseline: dict, channel: str) -> Optional[dict]:
        return baseline.get(channel)
