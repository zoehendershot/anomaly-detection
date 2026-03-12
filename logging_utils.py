#!/usr/bin/env python3
import logging
from pathlib import Path


LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_FILE = LOG_DIR / "anomaly-api.log"


def configure_logging() -> None:
    """Configure root logging once with local file + console handlers."""
    root_logger = logging.getLogger()
    if getattr(configure_logging, "_configured", False):
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    configure_logging._configured = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)


def sync_log_file_to_s3(s3_client, bucket: str, s3_key: str = "logs/anomaly-api.log", logger=None) -> None:
    """Upload the local application log file to S3."""
    active_logger = logger or get_logger(__name__)

    if not LOG_FILE.exists():
        active_logger.warning("Log file not found at %s; skipping S3 sync", LOG_FILE)
        return

    try:
        s3_client.upload_file(str(LOG_FILE), bucket, s3_key)
        active_logger.info("Uploaded log file to s3://%s/%s", bucket, s3_key)
    except Exception as exc:
        active_logger.error("Failed to upload log file to S3: %s", exc, exc_info=True)
