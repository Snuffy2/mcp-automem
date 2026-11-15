"""Behavioral tests for the Hermes provider's built-in memory mirror.

The provider template imports Hermes modules (``agent.memory_provider``,
``tools.registry``) that only exist inside a Hermes install, so they are stubbed
here. HTTP is faked at ``urllib.request.urlopen``: the provider's real
``AutoMemClient`` builds and sends every request.

Run with ``python3 -m unittest discover -s tests/hermes-provider`` (vitest runs it
through ``tests/hermes-provider/on-memory-write.test.ts``).
"""

from __future__ import annotations

import importlib.util
import io
import json
import logging
import pathlib
import sys
import threading
import time
import types
import unittest
import urllib.error
from typing import Any, Callable, Dict, List, Optional
from unittest import mock

sys.dont_write_bytecode = True

PROVIDER_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "templates" / "hermes" / "provider" / "__init__.py"
)


def _load_provider_module() -> types.ModuleType:
    memory = types.ModuleType("agent.memory_provider")
    memory.MemoryProvider = object
    registry = types.ModuleType("tools.registry")
    registry.tool_error = lambda message: message
    sys.modules.setdefault("agent", types.ModuleType("agent"))
    sys.modules["agent.memory_provider"] = memory
    sys.modules.setdefault("tools", types.ModuleType("tools"))
    sys.modules["tools.registry"] = registry
    sys.path.insert(0, str(PROVIDER_PATH.parent))
    spec = importlib.util.spec_from_file_location("automem_hermes_provider_under_test", PROVIDER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    module.__package__ = ""
    spec.loader.exec_module(module)
    return module


provider_module = _load_provider_module()


class _Response(io.BytesIO):
    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class FakeAutoMem:
    """Stands in for ``urlopen``; records requests and plays scripted outcomes."""

    def __init__(self, outcome: Optional[Callable[[], Any]] = None) -> None:
        self.requests: List[Dict[str, Any]] = []
        self.timeouts: List[float] = []
        self._outcome = outcome
        self.called = threading.Event()

    def __call__(self, request: Any, timeout: float) -> _Response:
        self.timeouts.append(timeout)
        self.requests.append(
            {
                "method": request.get_method(),
                "url": request.full_url,
                "headers": dict(request.header_items()),
                "body": json.loads(request.data.decode("utf-8")) if request.data else None,
            }
        )
        self.called.set()
        if self._outcome is not None:
            result = self._outcome()
            if isinstance(result, BaseException):
                raise result
        return _Response(json.dumps({"status": "success", "memory_id": "m-1"}).encode("utf-8"))


ENV = {
    "AUTOMEM_API_URL": "http://automem.test:8001",
    "AUTOMEM_API_KEY": "test-key",
}


class OnMemoryWriteTest(unittest.TestCase):
    def setUp(self) -> None:
        env = mock.patch.dict("os.environ", ENV, clear=False)
        env.start()
        self.addCleanup(env.stop)
        for key in ("AUTOMEM_HERMES_MIRROR_BUILTIN", "AUTOMEM_HERMES_DEBUG"):
            mock.patch.dict("os.environ", {key: ""}).start()
        self.addCleanup(mock.patch.stopall)

    def _provider(
        self, fake: FakeAutoMem, *, agent_context: str = "primary"
    ) -> Any:
        mock.patch.object(provider_module.urllib.request, "urlopen", fake).start()
        provider = provider_module.AutoMemMemoryProvider()
        provider.initialize("session-1", agent_context=agent_context)
        self.addCleanup(provider.shutdown)
        return provider

    def _write(self, provider: Any, *args: Any, **kwargs: Any) -> None:
        provider.on_memory_write(*args, **kwargs)
        provider.shutdown()  # drains the mirror queue before asserting

    def test_add_to_memory_stores_entry_with_provenance(self) -> None:
        fake = FakeAutoMem()
        provider = self._provider(fake)
        self._write(
            provider,
            "add",
            "memory",
            "  Deploys run from the release branch.  ",
            metadata={
                "write_origin": "assistant_tool",
                "session_id": "session-1",
                "tool_name": "memory",
                "platform": "cli",
                "unexpected": {"nested": True},
            },
        )

        self.assertEqual(len(fake.requests), 1)
        request = fake.requests[0]
        self.assertEqual(request["method"], "POST")
        self.assertEqual(request["url"], "http://automem.test:8001/memory")
        self.assertEqual(request["headers"].get("Authorization"), "Bearer test-key")
        self.assertEqual(
            request["body"],
            {
                "content": "Deploys run from the release branch.",
                "metadata": {
                    "source": "hermes_builtin_memory",
                    "hermes_action": "add",
                    "hermes_target": "memory",
                    "write_origin": "assistant_tool",
                    "session_id": "session-1",
                    "platform": "cli",
                },
            },
        )
        self.assertEqual(fake.timeouts, [provider_module.MIRROR_TIMEOUT])

    def test_add_to_user_profile_is_a_preference(self) -> None:
        fake = FakeAutoMem()
        provider = self._provider(fake)
        self._write(provider, "add", "user", "Prefers concise answers.")

        body = fake.requests[0]["body"]
        self.assertEqual(body["tags"], ["preference"])
        self.assertEqual(body["type"], "Preference")
        self.assertEqual(body["metadata"]["hermes_target"], "user")

    def test_replace_stores_new_content_and_records_previous(self) -> None:
        fake = FakeAutoMem()
        provider = self._provider(fake)
        self._write(
            provider,
            "replace",
            "memory",
            "Staging runs on port 7342.",
            {"previous_content": "Staging runs on port 7341.", "old_text": "7341"},
        )

        self.assertEqual(len(fake.requests), 1)
        body = fake.requests[0]["body"]
        self.assertEqual(body["content"], "Staging runs on port 7342.")
        self.assertEqual(body["metadata"]["hermes_action"], "replace")
        self.assertEqual(body["metadata"]["previous_content"], "Staging runs on port 7341.")
        self.assertNotIn("old_text", body["metadata"])

    def test_replace_without_change_is_not_mirrored(self) -> None:
        fake = FakeAutoMem()
        provider = self._provider(fake)
        self._write(
            provider,
            "replace",
            "memory",
            "Same entry.",
            metadata={"previous_content": " Same entry. "},
        )
        self.assertEqual(fake.requests, [])

    def test_remove_is_never_sent_to_automem(self) -> None:
        fake = FakeAutoMem()
        provider = self._provider(fake)
        self._write(
            provider,
            "remove",
            "memory",
            "",
            metadata={"previous_content": "Old fact."},
        )
        self.assertEqual(fake.requests, [])

    def test_empty_and_oversized_content_are_skipped(self) -> None:
        fake = FakeAutoMem()
        provider = self._provider(fake)
        provider.on_memory_write("add", "memory", "   ")
        provider.on_memory_write("add", "memory", "x" * (provider_module.MIRROR_MAX_CONTENT_CHARS + 1))
        provider.shutdown()
        self.assertEqual(fake.requests, [])

    def test_legacy_three_argument_call_still_mirrors(self) -> None:
        fake = FakeAutoMem()
        provider = self._provider(fake)
        self._write(provider, "add", "memory", "Older Hermes builds pass no metadata.")
        self.assertEqual(
            fake.requests[0]["body"]["metadata"],
            {"source": "hermes_builtin_memory", "hermes_action": "add", "hermes_target": "memory"},
        )

    def test_opt_out_env_disables_mirroring(self) -> None:
        mock.patch.dict("os.environ", {"AUTOMEM_HERMES_MIRROR_BUILTIN": "false"}).start()
        fake = FakeAutoMem()
        provider = self._provider(fake)
        self._write(provider, "add", "memory", "Should stay local.")
        self.assertEqual(fake.requests, [])

    def test_non_primary_agent_contexts_do_not_mirror(self) -> None:
        for context in ("cron", "flush", "subagent"):
            with self.subTest(context=context):
                fake = FakeAutoMem()
                provider = self._provider(fake, agent_context=context)
                self._write(provider, "add", "memory", "Written by a non-primary agent.")
                self.assertEqual(fake.requests, [])

    def test_write_returns_before_the_store_completes(self) -> None:
        release = threading.Event()
        fake = FakeAutoMem(outcome=lambda: release.wait(5) and None)
        provider = self._provider(fake)
        started = time.monotonic()
        provider.on_memory_write("add", "memory", "Slow AutoMem must not block the turn.")
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertTrue(fake.called.wait(2))
        release.set()
        provider.shutdown()
        self.assertEqual(len(fake.requests), 1)

    def test_failures_are_logged_once_and_never_retried_or_raised(self) -> None:
        outcomes = {
            "read timeout": TimeoutError("timed out"),
            "connect timeout": urllib.error.URLError(TimeoutError("timed out")),
            "connection refused": urllib.error.URLError(ConnectionRefusedError(61, "refused")),
            "http 500": urllib.error.HTTPError(
                "http://automem.test:8001/memory", 500, "Internal Server Error", None, None
            ),
        }
        for label, error in outcomes.items():
            with self.subTest(failure=label):
                fake = FakeAutoMem(outcome=lambda error=error: error)
                provider = self._provider(fake)
                with self.assertLogs(provider_module.logger, level=logging.WARNING) as logs:
                    self._write(provider, "add", "memory", f"Entry for {label}.")
                self.assertEqual(len(fake.requests), 1, "a failed store must not be retried")
                self.assertEqual(len(logs.records), 1)
                if "timeout" in label:
                    self.assertIn("not retried", logs.output[0])

    def test_full_queue_drops_the_write_without_raising(self) -> None:
        fake = FakeAutoMem()
        provider = self._provider(fake)
        provider._ensure_mirror_worker = lambda: None  # keep the queue undrained
        for index in range(provider_module.MIRROR_QUEUE_SIZE):
            provider.on_memory_write("add", "memory", f"Queued entry {index}.")
        with self.assertLogs(provider_module.logger, level=logging.WARNING) as logs:
            provider.on_memory_write("add", "memory", "One too many.")
        self.assertIn("queue full", logs.output[0])
        self.assertEqual(fake.requests, [])


if __name__ == "__main__":
    unittest.main()
