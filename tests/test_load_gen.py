"""Tests for the load generator's streaming parser, using a mock transport.

No real server or GPU: httpx.MockTransport returns a canned SSE stream so we can
assert that TTFT/ITL/token-count parsing is correct.
"""

import asyncio

import httpx

from bench.load_gen import make_prompt, one_request

_SSE_BODY = (
    'data: {"choices":[{"text":"Hello"}]}\n\n'
    'data: {"choices":[{"text":" world"}]}\n\n'
    'data: {"choices":[{"text":"!"}]}\n\n'
    'data: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":3}}\n\n'
    "data: [DONE]\n\n"
)


def _run(coro):
    return asyncio.run(coro)


def test_make_prompt_scales_with_tokens():
    short = make_prompt(8)
    long = make_prompt(256)
    assert len(long) > len(short)
    assert short  # non-empty


def test_one_request_parses_stream():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/completions"
        return httpx.Response(200, content=_SSE_BODY)

    transport = httpx.MockTransport(handler)

    async def run():
        async with httpx.AsyncClient(transport=transport) as client:
            return await one_request(client, "http://mock", "m", "prompt", 3)

    res = _run(run())
    assert res.success
    assert res.prompt_tokens == 7
    assert res.output_tokens == 3
    assert res.ttft > 0.0
    # 3 text chunks => first is TTFT, remaining 2 are inter-token gaps
    assert len(res.itls) == 2
    assert res.error is None


def test_one_request_records_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content="boom")

    transport = httpx.MockTransport(handler)

    async def run():
        async with httpx.AsyncClient(transport=transport) as client:
            return await one_request(client, "http://mock", "m", "prompt", 3)

    res = _run(run())
    assert not res.success
    assert res.error is not None
