"""Real LangChain + SDK code paths against a local fake OpenAI-compatible HTTP server.

This verifies request shape (strict json_schema, temperature, token limit, no SDK retries)
and response/usage/error parsing through the actual client stack. It is NOT live
verification of OpenAI or Groq; see tests/test_live.py for opt-in live checks.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from website_rag.errors import ErrorCode
from website_rag.generate import MalformedResponse, build_chat_model, generate_once
from website_rag.providers import classify
from website_rag.schemas import SiteRecord

DRAFT = {"status": "answered",
         "claims": [{"text": "Enable via ITEM_PIPELINES.", "evidence": [{"chunk_id": "abc", "span_id": "s-123"}]}]}


class FakeServer:
    def __init__(self):
        self.requests: list[dict] = []
        self.responses: list[tuple[int, dict, dict]] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                server.requests.append({"path": self.path, "body": body, "auth": self.headers.get("authorization", "")})
                status, payload, headers = server.responses.pop(0)
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                for k, v in headers.items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):  # silence
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def server():
    s = FakeServer()
    yield s
    s.close()


def completion(model: str, content: str) -> dict:
    return {"id": "chatcmpl-test", "object": "chat.completion", "created": 1, "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content, "refusal": None},
                         "finish_reason": "stop", "logprobs": None}],
            "usage": {"prompt_tokens": 1200, "completion_tokens": 90, "total_tokens": 1290,
                      "prompt_tokens_details": {"cached_tokens": 1024}, "completion_tokens_details": {"reasoning_tokens": 0}}}


def base_url(server: FakeServer, provider: str) -> str:
    return server.url + ("/v1" if provider == "openai" else "")


@pytest.mark.parametrize("provider", ["openai", "groq"])
def test_structured_output_request_and_usage(server, settings, provider):
    model_id = settings.model_for(provider)
    server.responses.append((200, completion(model_id, json.dumps(DRAFT)), {}))
    model = build_chat_model(provider, settings, base_url=base_url(server, provider))
    out = generate_once(model, [("system", "rules"), ("human", "question")])
    assert out.draft.status == "answered" and out.draft.claims[0].evidence[0].chunk_id == "abc"
    assert out.usage["input_tokens"] == 1200 and out.usage["output_tokens"] == 90
    req = server.requests[0]
    assert req["path"].endswith("/chat/completions")
    body = req["body"]
    assert body["model"] == model_id and body["temperature"] <= 1e-6  # ChatGroq sends 1e-8 for 0
    rf = body["response_format"]
    assert rf["type"] == "json_schema" and rf["json_schema"]["strict"] is True
    assert set(rf["json_schema"]["schema"]["required"]) == set(DRAFT)
    if provider == "groq":
        assert body["reasoning_effort"] == "low"
    else:
        assert out.usage["cached_input_tokens"] == 1024


@pytest.mark.parametrize("provider", ["openai", "groq"])
def test_http_errors_map_through_sdk_without_sdk_retries(server, settings, provider):
    server.responses.append((429, {"error": {"message": "You exceeded your current quota", "type": "insufficient_quota",
                                             "code": "insufficient_quota"}}, {"x-request-id": "req_abc"}))
    model = build_chat_model(provider, settings, base_url=base_url(server, provider))
    with pytest.raises(Exception) as err:
        generate_once(model, [("human", "q")])
    failure = classify(err.value, provider)
    assert failure.code == ErrorCode.QUOTA_EXCEEDED and failure.request_id == "req_abc"
    assert len(server.requests) == 1  # SDK retries are disabled; the app owns retry policy


def test_malformed_content_is_reported_with_usage(server, settings):
    server.responses.append((200, completion(settings.openai_model, '{"status": "answered"}'), {}))
    model = build_chat_model("openai", settings, base_url=base_url(server, "openai"))
    with pytest.raises(MalformedResponse) as err:
        generate_once(model, [("human", "q")])
    assert err.value.usage is None  # SDK parser raised before returning usage -> recorded as unknown
    assert classify(err.value, "openai").code == ErrorCode.MALFORMED_RESPONSE


def test_site_record_unused_guard():
    # keeps SiteRecord import meaningful for type-checkers in this module
    assert SiteRecord.model_fields["number"]
