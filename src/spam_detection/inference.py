from __future__ import annotations

from pathlib import Path
from typing import Any

from spam_detection.config import DEFAULT_MODEL_PATH
from spam_detection.data_processing.pipeline import vectorize_input_text
from spam_detection.utils import load_pickle


def load_inference_artifact(model_path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    artifact = load_pickle(model_path)
    if "model" not in artifact or "vectorizer" not in artifact:
        raise ValueError("Artefact invalide: il manque le modele ou le vectorizer.")
    return artifact


def predict_text(text: str, model_path: str | Path = DEFAULT_MODEL_PATH) -> int:
    artifact = load_inference_artifact(model_path)
    model = artifact["model"]
    vectorizer = artifact["vectorizer"]

    features = vectorize_input_text(text, vectorizer)
    prediction = model.predict(features)[0]
    return int(prediction)


def predict_label(text: str, model_path: str | Path = DEFAULT_MODEL_PATH) -> str:
    artifact = load_inference_artifact(model_path)
    decoder = artifact.get("label_decoder", {0: "ham", 1: "spam"})
    predicted_class = predict_text(text=text, model_path=model_path)
    return decoder.get(predicted_class, "unknown")
