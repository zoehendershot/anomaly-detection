#!/usr/bin/env python3
import json
import io
import logging
import boto3
import pandas as pd
from datetime import datetime

from baseline import BaselineManager
from detector import AnomalyDetector

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


NUMERIC_COLS = ["temperature", "humidity", "pressure", "wind_speed"]  # students configure this

def process_file(bucket: str, key: str):
    logger.info(f"Processing: s3://{bucket}/{key}")
    print(f"Processing: s3://{bucket}/{key}")

    try:
        # 1. Download raw file
        try:
            response = s3.get_object(Bucket=bucket, Key=key)
            file_content = response["Body"].read()
            logger.info(f"Downloaded {len(file_content)} bytes from s3://{bucket}/{key}")
        except s3.exceptions.NoSuchKey:
            error_msg = f"File not found: s3://{bucket}/{key}"
            logger.error(error_msg)
            print(f"ERROR: {error_msg}")
            raise
        except Exception as e:
            error_msg = f"Failed to download file s3://{bucket}/{key}: {e}"
            logger.error(error_msg, exc_info=True)
            print(f"ERROR: {error_msg}")
            raise

        try:
            df = pd.read_csv(io.BytesIO(file_content))
            logger.info(f"Loaded {len(df)} rows, columns: {list(df.columns)}")
            print(f"  Loaded {len(df)} rows, columns: {list(df.columns)}")
        except pd.errors.EmptyDataError:
            error_msg = f"Empty CSV file: s3://{bucket}/{key}"
            logger.error(error_msg)
            print(f"ERROR: {error_msg}")
            raise
        except pd.errors.ParserError as e:
            error_msg = f"Failed to parse CSV file s3://{bucket}/{key}: {e}"
            logger.error(error_msg, exc_info=True)
            print(f"ERROR: {error_msg}")
            raise
        except Exception as e:
            error_msg = f"Unexpected error reading CSV: {e}"
            logger.error(error_msg, exc_info=True)
            print(f"ERROR: {error_msg}")
            raise

        # 2. Load current baseline
        try:
            baseline_mgr = BaselineManager(bucket=bucket)
            baseline = baseline_mgr.load()
            logger.info("Baseline loaded successfully")
        except Exception as e:
            error_msg = f"Failed to load baseline: {e}"
            logger.error(error_msg, exc_info=True)
            print(f"ERROR: {error_msg}")
            raise

        # 3. Update baseline with values from this batch BEFORE scoring
        #    (use only non-null values for each channel)
        try:
            baseline_updated = False
            for col in NUMERIC_COLS:
                if col in df.columns:
                    clean_values = df[col].dropna().tolist()
                    if clean_values:
                        old_count = baseline.get(col, {}).get("count", 0)
                        baseline = baseline_mgr.update(baseline, col, clean_values)
                        new_count = baseline.get(col, {}).get("count", 0)
                        if new_count > old_count:
                            baseline_updated = True
                            logger.info(f"EVENT: Baseline updated for channel '{col}': {old_count} -> {new_count} observations")
            if baseline_updated:
                logger.info("EVENT: Baseline updated with new values from file")
                print("EVENT: Baseline updated with new values from file")
        except Exception as e:
            error_msg = f"Failed to update baseline: {e}"
            logger.error(error_msg, exc_info=True)
            print(f"ERROR: {error_msg}")
            raise

        # 4. Run detection
        try:
            logger.info(f"EVENT: Starting anomaly detection calculations for {len(df)} rows")
            print(f"EVENT: Starting anomaly detection calculations for {len(df)} rows")
            detector = AnomalyDetector(z_threshold=3.0, contamination=0.05)
            scored_df = detector.run(df, NUMERIC_COLS, baseline, method="both")
            anomaly_count = int(scored_df["anomaly"].sum()) if "anomaly" in scored_df.columns else 0
            logger.info(f"EVENT: Detection calculations completed - {anomaly_count} anomalies detected out of {len(df)} rows")
            print(f"EVENT: Detection calculations completed - {anomaly_count} anomalies detected")
        except Exception as e:
            error_msg = f"Failed to run anomaly detection: {e}"
            logger.error(error_msg, exc_info=True)
            print(f"ERROR: {error_msg}")
            raise

        # 5. Write scored file to processed/ prefix
        output_key = key.replace("raw/", "processed/")
        try:
            csv_buffer = io.StringIO()
            scored_df.to_csv(csv_buffer, index=False)
            s3.put_object(
                Bucket=bucket,
                Key=output_key,
                Body=csv_buffer.getvalue(),
                ContentType="text/csv"
            )
            logger.info(f"Wrote processed file: s3://{bucket}/{output_key}")
        except Exception as e:
            error_msg = f"Failed to write processed file s3://{bucket}/{output_key}: {e}"
            logger.error(error_msg, exc_info=True)
            print(f"ERROR: {error_msg}")
            raise

        # 6. Save updated baseline back to S3
        try:
            baseline_mgr.save(baseline)
            logger.info("Baseline saved successfully")
        except Exception as e:
            error_msg = f"Failed to save baseline: {e}"
            logger.error(error_msg, exc_info=True)
            print(f"ERROR: {error_msg}")
            raise

        # 7. Build and return a processing summary
        try:
            # anomaly_count already computed above during detection
            summary = {
                "source_key": key,
                "output_key": output_key,
                "processed_at": datetime.utcnow().isoformat(),
                "total_rows": len(df),
                "anomaly_count": anomaly_count,
                "anomaly_rate": round(anomaly_count / len(df), 4) if len(df) > 0 else 0,
                "baseline_observation_counts": {
                    col: baseline.get(col, {}).get("count", 0) for col in NUMERIC_COLS
                }
            }

            # Write summary JSON alongside the processed file
            summary_key = output_key.replace(".csv", "_summary.json")
            s3.put_object(
                Bucket=bucket,
                Key=summary_key,
                Body=json.dumps(summary, indent=2),
                ContentType="application/json"
            )
            logger.info(f"Wrote summary file: s3://{bucket}/{summary_key}")
        except Exception as e:
            error_msg = f"Failed to create/write summary: {e}"
            logger.error(error_msg, exc_info=True)
            print(f"ERROR: {error_msg}")
            raise

        logger.info(f"Successfully processed: {anomaly_count}/{len(df)} anomalies flagged")
        print(f"  Done: {anomaly_count}/{len(df)} anomalies flagged")
        return summary

    except Exception as e:
        error_msg = f"Critical error processing file s3://{bucket}/{key}: {e}"
        logger.error(error_msg, exc_info=True)
        print(f"CRITICAL ERROR: {error_msg}")
        raise
