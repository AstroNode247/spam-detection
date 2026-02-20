from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score
from sklearn.naive_bayes import MultinomialNB

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spam_detection.inference import (  # noqa: E402
    load_inference_artifact,
    predict_label,
    predict_text,
)
from spam_detection.utils import save_pickle  # noqa: E402


class TestInference(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = Path(self.temp_dir.name) / "naive_bayes_test.pkl"
        self._create_test_artifact(self.model_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @staticmethod
    def _create_test_artifact(model_path: Path) -> None:
        train_texts = [
            "win cash now",
            "free gift claim now",
            "project meeting at 10am",
            "please review the report",
            "urgent call this number",
            "see you tomorrow morning",
        ]
        train_labels = [1, 1, 0, 0, 1, 0]

        vectorizer = CountVectorizer()
        x_train = vectorizer.fit_transform(train_texts)
        model = MultinomialNB(alpha=1.0)
        model.fit(x_train, train_labels)

        artifact = {
            "model": model,
            "vectorizer": vectorizer,
            "label_decoder": {0: "ham", 1: "spam"},
        }
        save_pickle(artifact, model_path)

    def test_load_inference_artifact_contains_model_and_vectorizer(self) -> None:
        artifact = load_inference_artifact(self.model_path)
        self.assertIn("model", artifact)
        self.assertIn("vectorizer", artifact)

    def test_predict_text_returns_binary_class(self) -> None:
        prediction = predict_text("free cash prize", model_path=self.model_path)
        self.assertIn(prediction, (0, 1))

    def test_predict_label_returns_spam_or_ham(self) -> None:
        label = predict_label("please call me", model_path=self.model_path)
        self.assertIn(label, ("spam", "ham"))

    def test_inference_precision_is_high_on_control_set(self) -> None:
        eval_texts = [
            "free cash now",
            "claim your free gift",
            "meeting moved to 2pm",
            "please send the document",
            "win urgent prize",
            "let us have lunch",
        ]
        y_true = [1, 1, 0, 0, 1, 0]
        y_pred = [predict_text(text, model_path=self.model_path) for text in eval_texts]

        precision = precision_score(y_true, y_pred, zero_division=0)
        self.assertGreaterEqual(
            precision,
            0.80,
            msg=f"Precision trop basse: {precision:.3f}",
        )


if __name__ == "__main__":
    unittest.main()
