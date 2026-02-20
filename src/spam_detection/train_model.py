from __future__ import annotations

from pathlib import Path
from typing import Any

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNBd

from spam_detection.config import DEFAULT_MODEL_PATH, NB_ALPHA
from spam_detection.data_processing.pipeline import prepare_training_matrices
from spam_detection.log import get_logger
from spam_detection.utils import save_pickle

logger = get_logger(__name__)


def train_and_save_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    logger.info("Debut du pipeline d'entrainement.")
    logger.info("Etape 1/6 - Chargement et preparation des donnees.")
    prepared_data = prepare_training_matrices()

    x_train = prepared_data["x_train_vectorized"]
    x_test = prepared_data["x_test_vectorized"]
    y_train = prepared_data["y_train"]
    y_test = prepared_data["y_test"]
    vectorizer = prepared_data["vectorizer"]
    logger.info(
        "Donnees pretes. x_train=%s x_test=%s y_train=%d y_test=%d",
        x_train.shape,
        x_test.shape,
        len(y_train),
        len(y_test),
    )

    logger.info("Etape 2/6 - Initialisation du modele MultinomialNB (alpha=%.3f).", NB_ALPHA)
    model = MultinomialNB(alpha=NB_ALPHA)
    logger.info("Etape 3/6 - Entrainement du modele.")
    model.fit(x_train, y_train)
    logger.info("Entrainement termine.")

    logger.info("Etape 4/6 - Prediction sur le jeu de test.")
    y_pred = model.predict(x_test)

    logger.info("Etape 5/6 - Calcul des metriques d'evaluation.")
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    artifact = {
        "model": model,
        "vectorizer": vectorizer,
        "label_decoder": {0: "ham", 1: "spam"},
        "metrics": metrics,
    }

    logger.info("Etape 6/6 - Sauvegarde de l'artefact modele.")
    saved_path = save_pickle(artifact, model_path)

    logger.info(
        "Metrics -> accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
    )
    logger.info("Modele sauvegarde: %s", saved_path)
    logger.info("Pipeline d'entrainement termine avec succes.")

    return {"model_path": str(saved_path), "metrics": metrics}


if __name__ == "__main__":
    train_and_save_model()
