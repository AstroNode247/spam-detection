from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

API_BASE_DIR = Path(__file__).resolve().parent
PROJECT_SRC_DIR = API_BASE_DIR.parent

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_PREFIX = os.getenv("API_PREFIX", "/api/v1")

API_LOG_DIR = Path(os.getenv("API_LOG_DIR", str(API_BASE_DIR / "logs")))
API_LOG_FILE = os.getenv("API_LOG_FILE", "api.log")
API_LOG_PATH = API_LOG_DIR / API_LOG_FILE

API_LOG_DIR.mkdir(parents=True, exist_ok=True)
