"""PRD-02 transport tests against a local standard-library HTTP stub."""

from __future__ import annotations

import json
import socket
import unittest
import warnings

from tests.stub_server import StubServer
from tinyagent.backends import LlamaCppBackend, OllamaBackend
from tinyagent.errors import BackendError


DEFAULT_STOP = ["\n\n", "\nQuery:", "\nReply:"]


class BackendTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server = StubServer()
        self.addCleanup(self.server.close)

    def _ollama(self, *, timeout_s: float = 60.0, num_ctx: int | None = None):
        return OllamaBackend(
            "test-model", host=self.server.url, timeout_s=timeout_s, num_ctx=num_ctx
        )

    def _llamacpp(self, *, timeout_s: float = 60.0, chat_template: bool = True):
        return LlamaCppBackend(
            base_url=self.server.url, timeout_s=timeout_s, chat_template=chat_template
        )

    def test_ollama_request_contract(self) -> None:
        self.server.set_response(200, {"response": "  raw\ntext  "})
        result = self._ollama().complete("prompt", stop=DEFAULT_STOP, max_tokens=96)
        request = self.server.requests[0]
        self.assertEqual(request["path"], "/api/generate")
        headers = {name.lower(): value for name, value in request["headers"].items()}
        self.assertEqual(headers["content-type"], "application/json")
        self.assertEqual(
            request["json"],
            {
                "model": "test-model",
                "prompt": "prompt",
                "stream": False,
                "raw": False,
                "options": {
                    "temperature": 0.0,
                    "top_k": 1,
                    "top_p": 1.0,
                    "seed": 42,
                    "num_predict": 96,
                    "stop": DEFAULT_STOP,
                },
            },
        )
        self.assertEqual(result.text, "  raw\ntext  ")
        self.assertEqual(result.backend, "ollama:test-model")
        self.assertEqual(len(self.server.requests), 1)

    def test_ollama_optional_context_and_token_counts(self) -> None:
        self.server.set_response(
            200,
            {"response": "ok", "prompt_eval_count": 5, "eval_count": 2},
        )
        result = self._ollama(num_ctx=2048).complete(
            "prompt", stop=(), max_tokens=3
        )
        self.assertEqual(self.server.requests[0]["json"]["options"]["num_ctx"], 2048)
        self.assertEqual((result.prompt_tokens, result.completion_tokens), (5, 2))

    def test_ollama_omits_context_and_invalid_token_counts(self) -> None:
        self.server.set_response(
            200,
            {"response": "ok", "prompt_eval_count": "5", "eval_count": True},
        )
        result = self._ollama().complete("prompt", stop=(), max_tokens=1)
        self.assertNotIn("num_ctx", self.server.requests[0]["json"]["options"])
        self.assertEqual((result.prompt_tokens, result.completion_tokens), (None, None))

    def test_llamacpp_chat_request_contract(self) -> None:
        self.server.set_response(
            200,
            {
                "choices": [{"message": {"role": "assistant", "content": " raw\ntext "}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 4},
            },
        )
        result = self._llamacpp().complete("prompt", stop=DEFAULT_STOP, max_tokens=96)
        request = self.server.requests[0]
        self.assertEqual(request["path"], "/v1/chat/completions")
        self.assertEqual(
            request["json"],
            {
                "messages": [{"role": "user", "content": "prompt"}],
                "max_tokens": 96,
                "stream": False,
                "temperature": 0.0,
                "top_k": 1,
                "top_p": 1.0,
                "seed": 42,
                "stop": DEFAULT_STOP,
                "cache_prompt": True,
            },
        )
        self.assertEqual((result.text, result.backend), (" raw\ntext ", "llamacpp"))
        self.assertEqual((result.prompt_tokens, result.completion_tokens), (7, 4))
        self.assertEqual(len(self.server.requests), 1)

    def test_llamacpp_raw_completion_contract(self) -> None:
        self.server.set_response(
            200,
            {"content": "unchanged", "tokens_evaluated": 7, "tokens_predicted": 4},
        )
        result = self._llamacpp(chat_template=False).complete(
            "prompt", stop=DEFAULT_STOP, max_tokens=96
        )
        request = self.server.requests[0]
        self.assertEqual(request["path"], "/completion")
        self.assertEqual(
            request["json"],
            {
                "prompt": "prompt",
                "stream": False,
                "temperature": 0.0,
                "top_k": 1,
                "top_p": 1.0,
                "seed": 42,
                "n_predict": 96,
                "stop": DEFAULT_STOP,
                "cache_prompt": True,
            },
        )
        self.assertEqual((result.text, result.backend), ("unchanged", "llamacpp"))
        self.assertEqual((result.prompt_tokens, result.completion_tokens), (7, 4))

    def test_constructor_and_call_validation(self) -> None:
        for model in ("", "  ", None):
            with self.subTest(model=model), self.assertRaises(ValueError):
                OllamaBackend(model)  # type: ignore[arg-type]
        backend = self._ollama()
        for max_tokens in (0, 513, 1.5, True):
            with self.subTest(max_tokens=max_tokens), self.assertRaises(ValueError):
                backend.complete("prompt", stop=(), max_tokens=max_tokens)
        with self.assertRaises(ValueError):
            backend.complete("prompt", stop=None, max_tokens=96)  # type: ignore[arg-type]
        self.assertEqual(self.server.requests, [])

    def test_sampling_has_no_public_override(self) -> None:
        self.assertNotIn("temperature", OllamaBackend.__init__.__annotations__)
        self.assertNotIn("temperature", LlamaCppBackend.__init__.__annotations__)
        self.assertNotIn("temperature", OllamaBackend.complete.__annotations__)
        self.assertNotIn("temperature", LlamaCppBackend.complete.__annotations__)

    def test_unreachable_is_typed_and_uses_one_attempt(self) -> None:
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        backend = OllamaBackend("test-model", host=f"http://127.0.0.1:{port}")
        with self.assertRaises(BackendError) as raised:
            backend.complete("secret prompt", stop=(), max_tokens=1)
        self.assertEqual(raised.exception.code, "E_BACKEND_UNREACHABLE")
        self.assertNotIn("secret prompt", raised.exception.detail)
        self.assertLessEqual(len(raised.exception.detail), 200)

    def test_timeout_is_typed(self) -> None:
        self.server.set_response(200, {"response": "late"}, delay_s=0.4)
        with self.assertRaises(BackendError) as raised:
            self._ollama(timeout_s=0.05).complete(
                "prompt", stop=(), max_tokens=1
            )
        self.assertEqual(raised.exception.code, "E_BACKEND_TIMEOUT")
        self.assertIn("0.05s", raised.exception.detail)
        self.assertEqual(len(self.server.requests), 1)

    def test_http_error_precedes_url_error_mapping(self) -> None:
        for count, status in enumerate((404, 500), 1):
            with self.subTest(status=status):
                self.server.set_response(status, "server says no")
                with self.assertRaises(BackendError) as raised:
                    self._ollama().complete("secret prompt", stop=(), max_tokens=1)
                self.assertEqual(raised.exception.code, "E_BACKEND_HTTP")
                self.assertIn(f"HTTP {status}", raised.exception.detail)
                self.assertNotIn("secret prompt", raised.exception.detail)
                self.assertEqual(len(self.server.requests), count)

    def test_ollama_error_field(self) -> None:
        self.server.set_response(
            200, {"error": "model not found: secret prompt " + "x" * 250}
        )
        with self.assertRaises(BackendError) as raised:
            self._ollama().complete("secret prompt", stop=(), max_tokens=1)
        self.assertEqual(raised.exception.code, "E_BACKEND_HTTP")
        self.assertNotIn("secret prompt", raised.exception.detail)
        self.assertLessEqual(len(raised.exception.detail), 200)

    def test_protocol_errors(self) -> None:
        cases = (
            (b"not json", self._ollama()),
            (b"[]", self._ollama()),
            (b'{"response": 2}', self._ollama()),
            (b'{"content": null}', self._llamacpp(chat_template=False)),
            (b'{"content": "text"}', self._llamacpp()),
            (b'{"choices": []}', self._llamacpp()),
            (b'{"choices": [{"message": {"content": null}}]}', self._llamacpp()),
            (b'{"choices": ["x"]}', self._llamacpp()),
        )
        for body, backend in cases:
            with self.subTest(body=body, backend=backend):
                self.server.set_response(200, body)
                with self.assertRaises(BackendError) as raised:
                    backend.complete("prompt", stop=(), max_tokens=1)
                self.assertEqual(raised.exception.code, "E_BACKEND_PROTOCOL")

    def test_fifty_malformed_responses_only_raise_backend_error(self) -> None:
        backend = self._ollama()
        for index in range(50):
            body = (b"\xff" * (index + 1)) if index % 2 else json.dumps(
                {"wrong": index}
            ).encode("utf-8")
            self.server.set_response(200, body)
            with self.subTest(index=index), self.assertRaises(BackendError):
                backend.complete("prompt", stop=(), max_tokens=1)
        self.assertEqual(len(self.server.requests), 50)

    def test_remote_host_warns_and_repr_does_not_hold_prompt(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            backend = OllamaBackend("test-model", host="http://example.invalid:11434")
        self.assertEqual(len(caught), 1)
        self.assertIn("non-loopback", str(caught[0].message))
        self.assertNotIn("prompt", repr(backend).lower())


if __name__ == "__main__":
    unittest.main()
