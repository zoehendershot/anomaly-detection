# app.py
import io
import json
import os
import logging
import boto3
import pandas as pd
import requests
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks, Request, HTTPException
from baseline import BaselineManager
from processor import process_file

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

app = FastAPI(title="Anomaly Detection Pipeline")

try:
    s3 = boto3.client("s3")
    BUCKET_NAME = os.environ["BUCKET_NAME"]
    logger.info(f"Application initialized with bucket: {BUCKET_NAME}")
except KeyError as e:
    logger.error(f"Missing required environment variable: {e}")
    raise
except Exception as e:
    logger.error(f"Failed to initialize application: {e}", exc_info=True)
    raise

# ── SNS subscription confirmation + message handler ──────────────────────────

@app.post("/notify")
async def handle_sns(request: Request, background_tasks: BackgroundTasks):
    try:
        body = await request.json()
        msg_type = request.headers.get("x-amz-sns-message-type")
        logger.info(f"Received SNS message type: {msg_type}")

        # SNS sends a SubscriptionConfirmation before it will deliver any messages.
        # Visiting the SubscribeURL confirms the subscription.
        if msg_type == "SubscriptionConfirmation":
            try:
                confirm_url = body.get("SubscribeURL")
                if not confirm_url:
                    logger.error("SubscriptionConfirmation missing SubscribeURL")
                    return {"status": "error", "message": "Missing SubscribeURL"}
                logger.info(f"Confirming SNS subscription: {confirm_url}")
                response = requests.get(confirm_url, timeout=10)
                response.raise_for_status()
                logger.info("SNS subscription confirmed successfully")
                return {"status": "confirmed"}
            except requests.RequestException as e:
                logger.error(f"Failed to confirm SNS subscription: {e}", exc_info=True)
                return {"status": "error", "message": f"Subscription confirmation failed: {str(e)}"}
            except Exception as e:
                logger.error(f"Unexpected error during subscription confirmation: {e}", exc_info=True)
                return {"status": "error", "message": f"Unexpected error: {str(e)}"}

        if msg_type == "Notification":
            try:
                # The SNS message body contains the S3 event as a JSON string
                message_str = body.get("Message")
                if not message_str:
                    logger.error("Notification message missing 'Message' field")
                    return {"status": "error", "message": "Missing Message field"}
                
                s3_event = json.loads(message_str)
                records = s3_event.get("Records", [])
                logger.info(f"Processing {len(records)} S3 event record(s)")
                
                for record in records:
                    try:
                        key = record["s3"]["object"]["key"]
                        logger.info(f"Processing S3 object: {key}")
                        if key.startswith("raw/") and key.endswith(".csv"):
                            logger.info(f"EVENT: New file arrived - s3://{BUCKET_NAME}/{key}")
                            print(f"EVENT: New file arrived - s3://{BUCKET_NAME}/{key}")
                            background_tasks.add_task(process_file, BUCKET_NAME, key)
                            logger.info(f"Queued processing task for: {key}")
                        else:
                            logger.debug(f"Skipping non-CSV or non-raw file: {key}")
                    except KeyError as e:
                        logger.error(f"Malformed S3 event record (missing key): {e}", exc_info=True)
                    except Exception as e:
                        logger.error(f"Error processing S3 event record: {e}", exc_info=True)

                return {"status": "ok"}
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse SNS message JSON: {e}", exc_info=True)
                return {"status": "error", "message": f"Invalid JSON in message: {str(e)}"}
            except Exception as e:
                logger.error(f"Unexpected error processing notification: {e}", exc_info=True)
                return {"status": "error", "message": f"Unexpected error: {str(e)}"}

        logger.warning(f"Unknown message type: {msg_type}")
        return {"status": "ok", "message": f"Unknown message type: {msg_type}"}
    except Exception as e:
        logger.error(f"Critical error in handle_sns: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# ── Query endpoints ───────────────────────────────────────────────────────────

@app.get("/anomalies/recent")
def get_recent_anomalies(limit: int = 50):
    """Return rows flagged as anomalies across the 10 most recent processed files."""
    try:
        logger.info(f"Fetching recent anomalies with limit: {limit}")
        paginator = s3.get_paginator("list_objects_v2")
        
        try:
            pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix="processed/")
        except Exception as e:
            logger.error(f"Failed to list S3 objects: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"S3 list operation failed: {str(e)}")

        keys = []
        try:
            for page in pages:
                for obj in page.get("Contents", []):
                    if obj["Key"].endswith(".csv"):
                        keys.append(obj["Key"])
            keys = sorted(keys, reverse=True)[:10]
            logger.info(f"Found {len(keys)} processed CSV files")
        except Exception as e:
            logger.error(f"Error processing S3 pagination results: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error processing file list: {str(e)}")

        all_anomalies = []
        for key in keys:
            try:
                response = s3.get_object(Bucket=BUCKET_NAME, Key=key)
                df = pd.read_csv(io.BytesIO(response["Body"].read()))
                if "anomaly" in df.columns:
                    flagged = df[df["anomaly"] == True].copy()
                    flagged["source_file"] = key
                    all_anomalies.append(flagged)
            except s3.exceptions.NoSuchKey:
                logger.warning(f"File not found in S3: {key}")
            except pd.errors.EmptyDataError:
                logger.warning(f"Empty CSV file: {key}")
            except pd.errors.ParserError as e:
                logger.error(f"Failed to parse CSV file {key}: {e}", exc_info=True)
            except Exception as e:
                logger.error(f"Error processing file {key}: {e}", exc_info=True)

        if not all_anomalies:
            logger.info("No anomalies found in processed files")
            return {"count": 0, "anomalies": []}

        try:
            combined = pd.concat(all_anomalies).head(limit)
            result = {"count": len(combined), "anomalies": combined.to_dict(orient="records")}
            logger.info(f"Returning {len(combined)} anomalies")
            return result
        except Exception as e:
            logger.error(f"Error combining anomaly dataframes: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error combining results: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Critical error in get_recent_anomalies: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/anomalies/summary")
def get_anomaly_summary():
    """Aggregate anomaly rates across all processed files using their summary JSONs."""
    try:
        logger.info("Fetching anomaly summary")
        paginator = s3.get_paginator("list_objects_v2")
        
        try:
            pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix="processed/")
        except Exception as e:
            logger.error(f"Failed to list S3 objects: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"S3 list operation failed: {str(e)}")

        summaries = []
        for page in pages:
            for obj in page.get("Contents", []):
                if obj["Key"].endswith("_summary.json"):
                    try:
                        response = s3.get_object(Bucket=BUCKET_NAME, Key=obj["Key"])
                        summary_data = json.loads(response["Body"].read())
                        summaries.append(summary_data)
                    except s3.exceptions.NoSuchKey:
                        logger.warning(f"Summary file not found: {obj['Key']}")
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse summary JSON {obj['Key']}: {e}", exc_info=True)
                    except Exception as e:
                        logger.error(f"Error loading summary {obj['Key']}: {e}", exc_info=True)

        if not summaries:
            logger.info("No summary files found")
            return {"message": "No processed files yet."}

        try:
            total_rows = sum(s.get("total_rows", 0) for s in summaries)
            total_anomalies = sum(s.get("anomaly_count", 0) for s in summaries)
            anomaly_rate = round(total_anomalies / total_rows, 4) if total_rows > 0 else 0
            
            result = {
                "files_processed": len(summaries),
                "total_rows_scored": total_rows,
                "total_anomalies": total_anomalies,
                "overall_anomaly_rate": anomaly_rate,
                "most_recent": sorted(summaries, key=lambda x: x.get("processed_at", ""), reverse=True)[:5],
            }
            logger.info(f"Summary: {len(summaries)} files, {total_anomalies}/{total_rows} anomalies")
            return result
        except Exception as e:
            logger.error(f"Error calculating summary statistics: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error calculating summary: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Critical error in get_anomaly_summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/baseline/current")
def get_current_baseline():
    """Show the current per-channel statistics the detector is working from."""
    try:
        logger.info("Fetching current baseline")
        baseline_mgr = BaselineManager(bucket=BUCKET_NAME)
        
        try:
            baseline = baseline_mgr.load()
        except Exception as e:
            logger.error(f"Failed to load baseline: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to load baseline: {str(e)}")

        channels = {}
        try:
            for channel, stats in baseline.items():
                if channel == "last_updated":
                    continue
                if not isinstance(stats, dict):
                    logger.warning(f"Invalid stats format for channel {channel}")
                    continue
                channels[channel] = {
                    "observations": stats.get("count", 0),
                    "mean": round(stats.get("mean", 0.0), 4),
                    "std": round(stats.get("std", 0.0), 4),
                    "baseline_mature": stats.get("count", 0) >= 30,
                }
        except Exception as e:
            logger.error(f"Error processing baseline channels: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error processing baseline: {str(e)}")

        result = {
            "last_updated": baseline.get("last_updated"),
            "channels": channels,
        }
        logger.info(f"Returning baseline with {len(channels)} channels")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Critical error in get_current_baseline: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health")
def health():
    try:
        # Verify S3 access
        try:
            s3.head_bucket(Bucket=BUCKET_NAME)
            bucket_accessible = True
        except Exception as e:
            logger.warning(f"Health check: S3 bucket not accessible: {e}")
            bucket_accessible = False
        
        return {
            "status": "ok" if bucket_accessible else "degraded",
            "bucket": BUCKET_NAME,
            "bucket_accessible": bucket_accessible,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in health check: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
