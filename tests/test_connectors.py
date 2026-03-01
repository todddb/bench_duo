import unittest
from unittest.mock import MagicMock, patch

import requests

from app.connectors.base import Connector, ConnectorError
from app.connectors.detector import detect_backend
from app.connectors.mlx import MLXConnector
from app.connectors.ollama import OllamaConnector
from app.connectors.tensorrt_llm import TensorRTConnector


class ConnectorContractTests(unittest.TestCase):
    def test_connector_is_abstract(self) -> None:
        with self.assertRaises(TypeError):
            Connector()  # type: ignore[abstract]


class OllamaConnectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.connector = OllamaConnector(host="localhost", port=11434, timeout=1.0)

    @patch("app.connectors.ollama.requests.get")
    def test_probe_uses_health_endpoint(self, mock_get: MagicMock) -> None:
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.content = b'{"status":"ok"}'
        response.json.return_value = {"status": "ok"}
        mock_get.return_value = response

        result = self.connector.probe()

        self.assertTrue(result["ok"])
        self.assertEqual(result["endpoint"], "/api/health")

    @patch("app.connectors.ollama.requests.get")
    def test_probe_falls_back_to_version_endpoint(self, mock_get: MagicMock) -> None:
        bad_response = requests.RequestException("no health")
        good_response = MagicMock()
        good_response.raise_for_status.return_value = None
        good_response.content = b'{"version":"0.1"}'
        good_response.json.return_value = {"version": "0.1"}
        mock_get.side_effect = [bad_response, good_response]

        result = self.connector.probe()

        self.assertEqual(result["endpoint"], "/version")

    @patch("app.connectors.ollama.requests.get")
    def test_list_models_returns_names(self, mock_get: MagicMock) -> None:
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"models": [{"name": "llama3"}, {"name": "mistral"}]}
        mock_get.return_value = response

        models = self.connector.list_models()

        self.assertEqual(models, ["llama3", "mistral"])

    @patch("app.connectors.ollama.requests.post")
    def test_chat_returns_assistant_text(self, mock_post: MagicMock) -> None:
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"message": {"content": "hello user"}}
        mock_post.return_value = response

        text = self.connector.chat(
            messages=[{"role": "user", "content": "hi"}],
            settings={"model": "llama3", "temperature": 0.1},
        )

        self.assertEqual(text, "hello user")

    def test_chat_requires_model(self) -> None:
        with self.assertRaises(ConnectorError):
            self.connector.chat(messages=[{"role": "user", "content": "hi"}], settings={})


class MLXConnectorTests(unittest.TestCase):
    @patch("app.connectors.mlx.import_module")
    def test_probe_imports_mlx(self, mock_import: MagicMock) -> None:
        connector = MLXConnector()

        result = connector.probe()

        self.assertTrue(result["ok"])
        mock_import.assert_called_once_with("mlx_lm")

    @patch("app.connectors.mlx.import_module")
    def test_chat_uses_mlx_api(self, mock_import: MagicMock) -> None:
        connector = MLXConnector()
        mlx_mod = MagicMock()
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "formatted"
        mlx_mod.load.return_value = ("model", tokenizer)
        mlx_mod.generate.return_value = "mlx answer"
        mock_import.return_value = mlx_mod

        out = connector.chat(
            messages=[{"role": "user", "content": "hello"}],
            settings={"model_name": "mlx-community/test", "temperature": 0.3},
        )

        self.assertEqual(out, "mlx answer")


class BackendDetectorTests(unittest.TestCase):
    @patch("app.connectors.detector.requests.get")
    def test_detect_backend_prefers_ollama(self, mock_get: MagicMock) -> None:
        ollama_response = MagicMock()
        ollama_response.status_code = 200
        ollama_response.json.return_value = {"models": [{"name": "llama3"}, {"name": "mistral"}]}
        mock_get.return_value = ollama_response

        result = detect_backend("127.0.0.1", 9001)

        self.assertEqual(result["backend"], "ollama")
        self.assertEqual(result["models"], ["llama3", "mistral"])

    @patch("app.connectors.detector.requests.get")
    def test_detect_backend_returns_none_when_unreachable(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = requests.RequestException("Connection refused")

        result = detect_backend("127.0.0.1", 9001)

        self.assertIsNone(result)


class TensorRTConnectorTests(unittest.TestCase):
    @patch("app.connectors.tensorrt_llm.import_module")
    def test_probe_imports_tensorrt(self, mock_import: MagicMock) -> None:
        connector = TensorRTConnector()

        result = connector.probe()

        self.assertTrue(result["ok"])
        mock_import.assert_called_once_with("tensorrt_llm")

    @patch("app.connectors.tensorrt_llm.import_module")
    def test_chat_returns_content(self, mock_import: MagicMock) -> None:
        connector = TensorRTConnector()
        trt_mod = MagicMock()
        trt_mod.LLM.return_value.chat.return_value = [{"message": {"content": "trt answer"}}]
        mock_import.return_value = trt_mod

        out = connector.chat(
            messages=[{"role": "user", "content": "hello"}],
            settings={"model": "my-model"},
        )

        self.assertEqual(out, "trt answer")


if __name__ == "__main__":
    unittest.main()
