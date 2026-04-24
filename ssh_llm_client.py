"""
SSH-based LLM client — Routes inference through GPU instance via SSH tunnel.

Provides an alternative to local LLM by connecting to a remote GPU server
running vLLM with Qwen model.
"""

import json
import logging
import os
import subprocess
import time
from typing import Optional

logger = logging.getLogger("ssh_llm_client")


class SSHLLMClient:
    """
    SSH-tunneled LLM client that forwards requests to a remote vLLM server.

    Uses SSH tunnel to connect to gpu02.cc.iitk.ac.in where vLLM is running.
    """

    def __init__(
        self,
        ssh_host: str = "gpu02.cc.iitk.ac.in",
        ssh_user: str = "sunrajp23",
        local_port: int = 8000,
        remote_port: int = 8000,
        ssh_key: Optional[str] = None,
        timeout: float = 60.0,
    ):
        """
        Initialize SSH LLM client.

        Args:
            ssh_host: Remote server hostname
            ssh_user: SSH username
            local_port: Local port to bind (clients connect here)
            remote_port: Port on remote server where vLLM listens
            ssh_key: Path to SSH private key (optional)
            timeout: Request timeout in seconds
        """
        self.ssh_host = ssh_host
        self.ssh_user = ssh_user
        self.local_port = local_port
        self.remote_port = remote_port
        self.ssh_key = ssh_key or os.environ.get("SSH_KEY_PATH", "~/.ssh/id_rsa")
        self.timeout = timeout
        self._connected = False

        # Build SSH command
        self.ssh_cmd = self._build_ssh_command()

    def _build_ssh_command(self) -> str:
        """Build the SSH localhost tunnel command."""
        ssh_key = self.ssh_key.expanduser()

        cmd = [
            "ssh",
            "-N",  # No remote command
            "-f",  # Background the献策
            "-L", f"{self.local_port}:localhost:{self.remote_port}",
            "-o", "StrictHostKeyChecking=yes",
            "-o", "UserKnownHostsFile=/dev/null",  # Skip host verification (for dev)
            f"{self.ssh_user}@{self.ssh_host}",
        ]

        # Add SSH key if provided
        if os.path.exists(ssh_key):
            cmd.extend(["-i", ssh_key])

        return " ".join(cmd)

    def connect(self) -> bool:
        """
        Establish SSH tunnel connection.

        Returns:
            True if connection successful, False otherwise
        """
        logger.info("[SSH]  Starting SSH tunnel connection...")
        try:
            # Start SSH background process
            proc = subprocess.Popen(
                self.ssh_cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Wait briefly for SSH to establish connection
            logger.info("[SSH]  Waiting for tunnel to establish...")
            for _ in range(30):  # Up to 30 seconds
                try:
                    time.sleep(1)
                    # Check if process is still running
                    if proc.poll() is not None:
                        stderr = proc.stderr.read().decode() if proc.stderr else ""
                        logger.error("[SSH]  SSH process terminated unexpectedly: %s", stderr)
                        return False
                except Exception as e:
                    logger.error("[SSH]  Error waiting for tunnel: %s", e)
                    return False

            logger.info("[SSH]  Tunnel established: %s:%d -> %s:%d",
                        self.ssh_host, self.remote_port, "localhost", self.local_port)
            self._connected = True
            return True

        except FileNotFoundError:
            logger.error("[SSH]  SSH command not found. Is SSH installed?")
            return False
        except Exception as e:
            logger.error("[SSH]  Connection failed: %s", e)
            return False

    def disconnect(self):
        """Close SSH tunnel connection."""
        if hasattr(self, "_proc") and self._proc.poll() is None:
            self._proc.terminate()
            logger.info("[SSH]  SSH tunnel closed")
        self._connected = False

    def is_connected(self) -> bool:
        """Check if SSH tunnel is connected."""
        return self._connected

    def _make_request(self, prompt: str, temperature: float = 0.0) -> str:
        """
        Send request to vLLM API via local port.

        Uses HTTP POST to http://localhost:{local_port}/v1/chat/completions
        """
        url = f"http://localhost:{self.local_port}/v1/chat/completions"

        payload = {
            "model": "Qwen2.5-32B-Instruct",  # Default model on vLLM
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": 4096,
        }

        try:
            import requests
        except ImportError:
            import urllib.parse
            import urllib.request

            params = urllib.parse.urlencode(payload)
            req = urllib.request.Request(url, data=params.encode("utf-8"),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"]

        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]

    def invoke(self, prompt: str, temperature: float = 0.0) -> str:
        """
        Invoke LLM inference on remote GPU server.

        Args:
            prompt: Input prompt/text to process
            temperature: Sampling temperature (0.0 = deterministic)

        Returns:
            LLM response text
        """
        if not self._connected:
            raise RuntimeError("SSH tunnel not connected. Call connect() first.")

        # For structured outputs, wrap in chain-of-thought format
        if "{" in prompt or "JSON" in prompt or "json" in prompt.lower():
            thought_prompt = f"""Analyze this request and output ONLY the requested JSON structure.
If asked to classify/query data about perovskite solar cells, use one of:
DATA_QUERY, DATA_PLUS_EXPLANATION, DOMAIN_ONLY, DESIGN, MATERIAL_LOOKUP,
PROPERTY_PREDICT, MULTI_STEP, DISAMBIGUATION_REQUIRED

Query: {prompt}
Reply with ONLY the category name."""
        else:
            thought_prompt = prompt

        response = self._make_request(thought_prompt, temperature)
        return response

    def __call__(self, prompt: str, temperature: float = 0.0) -> str:
        """Allow calling instance directly like a function."""
        return self.invoke(prompt, temperature)

    def pretty_print(self):
        """Print connection status in a nice format."""
        status = "CONNECTED" if self.is_connected() else "DISCONNECTED"
        print(f"\n{'='*50}")
        print(f"SSH LLM Client Status: {status}")
        print(f"  Host: {self.ssh_host}")
        print(f"  Remote Port: {self.remote_port}")
        print(f"  Local Port: {self.local_port}")
        print(f"  Model: Qwen2.5-32B-Instruct")
        print(f"{'='*50}\n")
