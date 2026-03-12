#!/usr/bin/env python3
import json
import math
import logging
import os
import boto3
from datetime import datetime
from typing import Optional

# Configure logging to both console and file
log_file = "/var/log/anomaly-api.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

s3 = boto3.client("s3")


def sync_log_to_s3(bucket: str, log_file_path: str = "/var/log/anomaly-api.log"):
    """
    Sync the application log file to S3 whenever baseline is saved.
    Uploads to logs/anomaly-api-{timestamp}.log
    """
    try:
        if not os.path.exists(log_file_path):
            logger.warning(f"Log file does not exist: {log_file_path}")
            return
        
        # Read current log file
        with open(log_file_path, 'r') as f:
            log_content = f.read()
        
        if not log_content.strip():
            logger.debug("Log file is empty, skipping sync")
            return
        
        # Create timestamped log key
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        log_key = f"logs/anomaly-api-{timestamp}.log"
        
        # Upload to S3
        s3.put_object(
            Bucket=bucket,
            Key=log_key,
            Body=log_content,
            ContentType="text/plain"
        )
        logger.info(f"Synced log file to s3://{bucket}/{log_key}")
    except PermissionError:
        logger.warning(f"Permission denied reading log file: {log_file_path}")
    except Exception as e:
        logger.error(f"Failed to sync log file to S3: {e}", exc_info=True)


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

        # copy of final baseline
        s3.put_object(
            Bucket=self.bucket,
            Key="state/baseline-final.json",
            Body=baseline_json,
            ContentType="application/json"
        )
        logger.info(f"Saved final baseline copy to s3://{self.bucket}/state/baseline-final.json")

        # sync log file too
        log_path = "/var/log/anomaly-api.log"
        if os.path.exists(log_path):
            s3.upload_file(log_path, self.bucket, "logs/anomaly-api.log")
            logger.info(f"Uploaded log file to s3://{self.bucket}/logs/anomaly-api.log")
        else:
            logger.warning("Log file not found, skipping log upload")


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
