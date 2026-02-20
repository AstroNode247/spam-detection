from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fastapi.testclient import TestClient  # noqa: E402

from api.config import API_PREFIX  # noqa: E402
from api.main import create_app  # noqa: E402


class TestApiEndpoints(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(create_app())

    def test_health_endpoint_returns_ok(self) -> None:
        response = self.client.get(f"{API_PREFIX}/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    @patch("api.main._run_model_inference", return_value=(1, "spam"))
    def test_predict_endpoint_returns_expected_payload(
        self,
        mock_inference,
    ) -> None:
        response = self.client.post(
            f"{API_PREFIX}/predict",
            json={"text": "free cash now"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["prediction"], 1)
        self.assertEqual(payload["label"], "spam")
        self.assertEqual(payload["input_text"], "free cash now")
        mock_inference.assert_called_once_with("free cash now")


if __name__ == "__main__":
    unittest.main()
