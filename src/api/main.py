from __future__ import annotations

from typing import Any

from api.config import API_HOST, API_PORT, API_PREFIX
from api.log import get_api_logger

from fastapi import FastAPI, HTTPException
from api.schema import InferenceRequest, InferenceResponse


logger = get_api_logger()

def predict_from_text(text: str) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Le texte d'entree ne doit pas etre vide.")

    logger.info("Inference API - reception d'une requete.")
    prediction, label = _run_model_inference(text)
    logger.info("Inference API - prediction terminee. class=%d label=%s", prediction, label)

    return {
        "input_text": text,
        "prediction": prediction,
        "label": label,
    }


def _run_model_inference(text: str) -> tuple[int, str]:
    # Import local pour decoupler le chargement de l'API des dependances ML lourdes.
    from spam_detection.inference import predict_label, predict_text

    prediction = predict_text(text=text)
    label = predict_label(text=text)
    return prediction, label


app = FastAPI(title="Spam Detection API", version="1.0.0")

@app.get(f"{API_PREFIX}/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post(f"{API_PREFIX}/predict", response_model=InferenceResponse)
def predict_endpoint(payload: InferenceRequest) -> InferenceResponse:
    try:
        return predict_from_text(payload.text)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        logger.exception("Erreur API pendant l'inference")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur interne lors de l'inference: {error}",
        ) from error

def create_app() -> FastAPI:
    return app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(create_app(), host=API_HOST, port=API_PORT)
