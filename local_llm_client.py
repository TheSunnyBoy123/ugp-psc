"""
Local LLM Client — Direct HTTP calls to vLLM running on localhost:8000

Usage:
    vLLM: python -m vllm.entrypoints.openai.api_server --model /scratch/work/sunrajp23/hf_cache/qwen-coder-3.5 --port 8000

Then in the chatbot:
    Toggle "Use Local Model" to route inference to localhost:8000/v1/chat/completions
"""

import json
import logging
import urllib.error
import urllib.request
from typing import Any, List, Optional

logger = logging.getLogger("local_llm_client")


class LocalLLMClient:
    """
    Simple HTTP client for vLLM local serving.

    Sends prompts to: http://localhost:{port}/v1/chat/completions

    Example vLLM startup:
        vllm serve /scratch/work/sunrajp23/hf_cache/qwen-coder-3.5 --port 8000 --model-max-len 8192
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        port: int = 8000,
        model_name: Optional[str] = None,
        timeout: float = 120.0,
    ):
        """
        Initialize local LLM client.

        Args:
            port: vLLM server port (default: 8000)
            model_name: Model name for API request
            timeout: Request timeout in seconds
        """
        self.base_url = (base_url or f"http://localhost:{port}/v1").rstrip("/")
        self.url = f"{self.base_url}/chat/completions"
        self.models_url = f"{self.base_url}/models"
        self.port = port
        self.model_name = model_name
        self.timeout = timeout
        logger.info("[LLM]  Local LLM client initialized — port=%d  model=%s  timeout=%.1fs",
                    port, model_name or "<auto>", timeout)

    def _request_json(self, url: str, payload: Optional[dict] = None) -> Any:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def list_models(self) -> List[str]:
        """Return model ids advertised by the local OpenAI-compatible server."""
        try:
            result = self._request_json(self.models_url)
        except Exception as exc:
            logger.warning("[LLM]  Failed to discover local models: %s", exc)
            return []

        models = []
        for item in result.get("data", []):
            model_id = item.get("id")
            if model_id:
                models.append(model_id)
        return models

    def is_available(self) -> bool:
        """Check whether the local server is reachable."""
        return bool(self.list_models())

    def resolve_model_name(self) -> str:
        """Pick the configured model or discover the first available model."""
        if self.model_name:
            return self.model_name

        models = self.list_models()
        if not models:
            raise RuntimeError(
                f"Cannot connect to local LLM server at {self.base_url}. "
                "Make sure the OpenAI-compatible server is running."
            )

        self.model_name = models[0]
        logger.info("[LLM]  Auto-selected local model: %s", self.model_name)
        return self.model_name

    def invoke(self, prompt: str, temperature: float = 0.0) -> str:
        """
        Invoke inference via vLLM HTTP API.

        Args:
            prompt: Input prompt text
            temperature: Sampling temperature (0.0 = deterministic)

        Returns:
            LLM response text
        """
        payload = {
            "model": self.resolve_model_name(),
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": 8192,
        }

        try:
            result = self._request_json(self.url, payload)
            return result["choices"][0]["message"]["content"]

        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            logger.error("[LLM]  HTTP error %d: %s", e.code, error_body[:200])
            if e.code == 503:
                raise RuntimeError(
                    "Local LLM server is not ready. Start the OpenAI-compatible server on localhost:8000."
                )
            raise RuntimeError(f"vLLM API error {e.code}: {error_body}")

        except urllib.error.URLError as e:
            logger.error("[LLM]  URL error: %s", e.reason)
            raise RuntimeError(
                f"Cannot connect to {self.base_url}. Is the local LLM server running?"
            )

        except json.JSONDecodeError as e:
            logger.error("[LLM]  JSON parse error: %s", e)
            raise RuntimeError(f"Invalid response from vLLM: {e}")

        except Exception as e:
            logger.error("[LLM]  Unexpected error: %s", e)
            raise RuntimeError(f"Local LLM inference failed: {e}")

    def pretty_print(self):
        """Print connection details."""
        print(f"\n{'='*50}")
        print(f"Local LLM Client Configuration:")
        print(f"  URL: {self.url}")
        print(f"  Models URL: {self.models_url}")
        print(f"  Model: {self.model_name or '<auto>'}")
        print(f"  Timeout: {self.timeout}s")
        print(f"{'='*50}\n")
