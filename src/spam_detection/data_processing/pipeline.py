from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from spam_detection.config import (
    DATA_FILE_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    VECTORIZER_MAX_FEATURES,
)
from spam_detection.data_processing.feature import (
    build_vectorizer,
    clean_text,
    clean_text_series,
)


def load_dataset(data_path: str | Path = DATA_FILE_PATH) -> pd.DataFrame:
    dataset_path = Path(data_path)
    return pd.read_table(
        dataset_path,
        sep="\t",
        names=["label", "sms_message"],
    )


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    encoded_df = df.copy()
    encoded_df["label"] = encoded_df["label"].map({"spam": 1, "ham": 0})
    if encoded_df["label"].isna().any():
        raise ValueError("Le dataset contient des labels inattendus.")
    encoded_df["label"] = encoded_df["label"].astype(int)
    return encoded_df


def prepare_features_and_target(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    x_text = clean_text_series(df["sms_message"])
    y = df["label"]
    return x_text, y


def split_data(
    x_text: pd.Series,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    return train_test_split(
        x_text,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def vectorize_train_test(
    x_train: pd.Series,
    x_test: pd.Series,
    vectorizer: CountVectorizer | None = None,
) -> tuple[CountVectorizer, spmatrix, spmatrix]:
    used_vectorizer = vectorizer or build_vectorizer(
        max_features=VECTORIZER_MAX_FEATURES
    )
    x_train_vectorized = used_vectorizer.fit_transform(x_train)
    x_test_vectorized = used_vectorizer.transform(x_test)
    return used_vectorizer, x_train_vectorized, x_test_vectorized


def prepare_training_matrices(
    data_path: str | Path = DATA_FILE_PATH,
) -> dict[str, Any]:
    df = load_dataset(data_path)
    df = encode_labels(df)
    x_text, y = prepare_features_and_target(df)
    x_train, x_test, y_train, y_test = split_data(x_text, y)
    vectorizer, x_train_vectorized, x_test_vectorized = vectorize_train_test(
        x_train, x_test
    )
    return {
        "x_train_vectorized": x_train_vectorized,
        "x_test_vectorized": x_test_vectorized,
        "y_train": y_train,
        "y_test": y_test,
        "vectorizer": vectorizer,
    }


def preprocess_input_text(text: object) -> str:
    return clean_text(text)


def vectorize_input_text(
    text: object,
    vectorizer: CountVectorizer,
) -> spmatrix:
    processed_text = preprocess_input_text(text)
    return vectorizer.transform([processed_text])
