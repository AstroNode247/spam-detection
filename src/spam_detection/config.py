from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("SPAM_DATA_DIR", str(PACKAGE_DIR / "data")))
MODEL_DIR = Path(os.getenv("SPAM_MODEL_DIR", str(PACKAGE_DIR / "model")))
LOG_DIR = Path(os.getenv("SPAM_LOG_DIR", str(PACKAGE_DIR / "logs")))

MODEL_FILENAME = os.getenv("SPAM_MODEL_FILENAME", "naive_bayes_spam.pkl")
DEFAULT_MODEL_PATH = MODEL_DIR / MODEL_FILENAME
DATA_FILE_PATH = DATA_DIR / "smsspamcollection" / "SMSSpamCollection"

LOG_FILE_NAME = os.getenv("SPAM_LOG_FILE", "spam_detection.log")
LOG_FILE_PATH = LOG_DIR / LOG_FILE_NAME

TEST_SIZE = float(os.getenv("SPAM_TEST_SIZE", "0.25"))
RANDOM_STATE = int(os.getenv("SPAM_RANDOM_STATE", "1"))
NB_ALPHA = float(os.getenv("SPAM_NB_ALPHA", "1.0"))
VECTORIZER_MAX_FEATURES = (
    int(os.getenv("SPAM_MAX_FEATURES"))
    if os.getenv("SPAM_MAX_FEATURES")
    else None
)

# Création des dossiers runtime dès l'import.
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
