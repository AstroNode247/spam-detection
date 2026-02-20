from __future__ import annotations

import re

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

_NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9\s]+")
_MULTI_SPACE_PATTERN = re.compile(r"\s+")


def to_text(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def normalize_case_and_spaces(text: str) -> str:
    lowered = text.lower().strip()
    return _MULTI_SPACE_PATTERN.sub(" ", lowered)


def remove_special_characters(text: str) -> str:
    return _NON_ALNUM_PATTERN.sub(" ", text)


def clean_text(text: object) -> str:
    value = to_text(text)
    value = normalize_case_and_spaces(value)
    value = remove_special_characters(value)
    value = _MULTI_SPACE_PATTERN.sub(" ", value)
    return value.strip()


def clean_text_series(text_series: pd.Series) -> pd.Series:
    return text_series.fillna("").apply(clean_text)


def build_vectorizer(max_features: int | None = None) -> CountVectorizer:
    return CountVectorizer(max_features=max_features)
