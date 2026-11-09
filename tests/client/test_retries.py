from __future__ import annotations

import datetime
import email.utils
import typing

import anyio
import pytest

import httpx


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """
    Record the delays passed to `time.sleep` and `anyio.sleep`, without waiting.
    """
    calls: list[float] = []

    def fake_sleep(seconds: float) -> None:
        calls.append(seconds)

    async def fake_async_sleep(seconds: float) -> None:
        calls.append(seconds)

    monkeypatch.setattr("time.sleep", fake_sleep)
    monkeypatch.setattr(anyio, "sleep", fake_async_sleep)
    return calls


class Server:
    """
    A mock server that returns a sequence of responses (or raises exceptions),
    recording every request it receives.
    """

    def __init__(self, *outcomes: int | httpx.Response | Exception) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        request.read()
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, int):
            return httpx.Response(outcome)
        return outcome

    @property
    def count(self) -> int:
        return len(self.requests)


class TrackedStream(httpx.SyncByteStream, httpx.AsyncByteStream):
    def __init__(self, closed: list[TrackedStream]) -> None:
        self.closed = closed

    def __iter__(self) -> typing.Iterator[bytes]:
        yield b"body"

    async def __aiter__(self) -> typing.AsyncIterator[bytes]:
        yield b"body"

    def close(self) -> None:
        self.closed.append(self)

    async def aclose(self) -> None:
        self.closed.append(self)


def http_date(delta: datetime.timedelta) -> str:
    when = datetime.datetime.now(datetime.timezone.utc) + delta
    return email.utils.format_datetime(when, usegmt=True)


def make_client(handler: typing.Any, **kwargs: typing.Any) -> httpx.Client:
    kwargs.setdefault("retry", httpx.RetryPolicy())
    return httpx.Client(transport=httpx.MockTransport(handler), **kwargs)


def make_async_client(handler: typing.Any, **kwargs: typing.Any) -> httpx.AsyncClient:
    kwargs.setdefault("retry", httpx.RetryPolicy())
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs)


# RetryPolicy configuration


def test_retry_policy_defaults() -> None:
    retry = httpx.RetryPolicy()
    assert retry.max_attempts == 3
    assert retry.backoff_factor == 0.5
    assert retry.max_backoff == 60.0
    assert retry.retry_status_codes == {429, 500, 502, 503, 504}
    assert retry.retry_methods == {"GET", "HEAD", "PUT", "DELETE", "OPTIONS", "TRACE"}
    assert retry.respect_retry_after is True
    assert retry == httpx.RetryPolicy()
    assert retry != httpx.RetryPolicy(max_attempts=5)
    assert repr(retry) == (
        "RetryPolicy(max_attempts=3, backoff_factor=0.5, max_backoff=60.0, "
        "retry_status_codes=[429, 500, 502, 503, 504], "
        "retry_methods=['DELETE', 'GET', 'HEAD', 'OPTIONS', 'PUT', 'TRACE'], "
        "respect_retry_after=True)"
    )


def test_retry_policy_normalizes_methods() -> None:
    retry = httpx.RetryPolicy(retry_methods={"post", "Get"})
    assert retry.retry_methods == {"POST", "GET"}


@pytest.mark.parametrize(
    "kwargs",
    [{"max_attempts": 0}, {"backoff_factor": -1}, {"max_backoff": -1}],
)
def test_retry_policy_invalid_values(kwargs: dict[str, typing.Any]) -> None:
    with pytest.raises(ValueError):
        httpx.RetryPolicy(**kwargs)


def test_client_retry_attribute() -> None:
    retry = httpx.RetryPolicy()
    assert httpx.Client().retry is None
    assert httpx.Client(retry=retry).retry is retry
    assert httpx.AsyncClient(retry=retry).retry is retry


# Status code retries


def test_retry_503_then_200(sleeps: list[float]) -> None:
    server = Server(503, 200)
    with make_client(server) as client:
        response = client.get("https://example.org/")

    assert response.status_code == 200
    assert server.count == 2
    assert len(sleeps) == 1


def test_retry_after_seconds(sleeps: list[float]) -> None:
    server = Server(httpx.Response(429, headers={"Retry-After": "1"}), 200)
    with make_client(server) as client:
        response = client.get("https://example.org/")

    assert response.status_code == 200
    assert server.count == 2
    assert sleeps == [1.0]


def test_retry_after_http_date_in_the_past(sleeps: list[float]) -> None:
    retry_after = http_date(-datetime.timedelta(hours=1))
    server = Server(httpx.Response(503, headers={"Retry-After": retry_after}), 200)
    with make_client(server) as client:
        response = client.get("https://example.org/")

    assert response.status_code == 200
    assert sleeps == [0.0]


def test_retry_after_http_date_in_the_future(sleeps: list[float]) -> None:
    retry_after = http_date(datetime.timedelta(seconds=30))
    server = Server(httpx.Response(503, headers={"Retry-After": retry_after}), 200)
    with make_client(server) as client:
        client.get("https://example.org/")

    assert len(sleeps) == 1
    assert 25 <= sleeps[0] <= 30


def test_retry_after_http_date_without_timezone(sleeps: list[float]) -> None:
    retry_after = "Sun, 06 Nov 1994 08:49:37 -0000"
    server = Server(httpx.Response(503, headers={"Retry-After": retry_after}), 200)
    with make_client(server) as client:
        client.get("https://example.org/")

    assert sleeps == [0.0]


def test_retry_after_capped_at_max_backoff(sleeps: list[float]) -> None:
    server = Server(httpx.Response(503, headers={"Retry-After": "120"}), 200)
    with make_client(server, retry=httpx.RetryPolicy(max_backoff=10)) as client:
        client.get("https://example.org/")

    assert sleeps == [10.0]


@pytest.mark.parametrize("value", ["invalid", "-5", "1.5", "", "\u00b2"])
def test_invalid_retry_after_falls_back_to_backoff(
    sleeps: list[float], monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setattr("random.uniform", lambda a, b: b)
    server = Server(httpx.Response(503, headers={b"Retry-After": value.encode()}), 200)
    with make_client(server) as client:
        client.get("https://example.org/")

    assert sleeps == [0.5]


def test_retry_after_ignored_when_disabled(
    sleeps: list[float], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("random.uniform", lambda a, b: b)
    server = Server(httpx.Response(503, headers={"Retry-After": "30"}), 200)
    retry = httpx.RetryPolicy(respect_retry_after=False)
    with make_client(server, retry=retry) as client:
        client.get("https://example.org/")

    assert sleeps == [0.5]


def test_backoff_formula(sleeps: list[float], monkeypatch: pytest.MonkeyPatch) -> None:
    jitter_bounds: list[tuple[float, float]] = []

    def fake_uniform(a: float, b: float) -> float:
        jitter_bounds.append((a, b))
        return b / 2

    monkeypatch.setattr("random.uniform", fake_uniform)
    server = Server(503)
    retry = httpx.RetryPolicy(max_attempts=6, backoff_factor=1.0, max_backoff=10.0)
    with make_client(server, retry=retry) as client:
        client.get("https://example.org/")

    assert jitter_bounds == [(0, 1.0), (0, 2.0), (0, 4.0), (0, 8.0), (0, 10.0)]
    assert sleeps == [0.5, 1.0, 2.0, 4.0, 5.0]


def test_post_503_not_retried_by_default(sleeps: list[float]) -> None:
    server = Server(503, 200)
    with make_client(server) as client:
        response = client.post("https://example.org/", content=b"data")

    assert response.status_code == 503
    assert server.count == 1
    assert sleeps == []


def test_post_retried_when_in_retry_methods(sleeps: list[float]) -> None:
    server = Server(503, 200)
    retry = httpx.RetryPolicy(retry_methods={"POST"})
    with make_client(server, retry=retry) as client:
        response = client.post("https://example.org/", content=b"data")

    assert response.status_code == 200
    assert [request.content for request in server.requests] == [b"data", b"data"]


def test_get_503_exhausts_max_attempts(sleeps: list[float]) -> None:
    server = Server(503)
    with make_client(server) as client:
        response = client.get("https://example.org/")

    assert response.status_code == 503
    assert response.is_closed
    assert response.content == b""
    assert server.count == 3
    assert len(sleeps) == 2


def test_max_attempts_one_disables_retries(sleeps: list[float]) -> None:
    server = Server(503, 200)
    with make_client(server, retry=httpx.RetryPolicy(max_attempts=1)) as client:
        response = client.get("https://example.org/")

    assert response.status_code == 503
    assert server.count == 1


@pytest.mark.parametrize("status_code", [200, 400, 404, 501])
def test_other_status_codes_not_retried(sleeps: list[float], status_code: int) -> None:
    server = Server(status_code, 200)
    with make_client(server) as client:
        response = client.get("https://example.org/")

    assert response.status_code == status_code
    assert server.count == 1


def test_custom_retry_status_codes(sleeps: list[float]) -> None:
    server = Server(404, 200)
    retry = httpx.RetryPolicy(retry_status_codes={404})
    with make_client(server, retry=retry) as client:
        response = client.get("https://example.org/")

    assert response.status_code == 200
    assert server.count == 2


# Exception retries


@pytest.mark.parametrize(
    "exc_class",
    [
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
        httpx.WriteTimeout,
        httpx.PoolTimeout,
        httpx.RemoteProtocolError,
    ],
)
def test_retryable_exception_then_success(
    sleeps: list[float], exc_class: type[httpx.TransportError]
) -> None:
    server = Server(exc_class("failed"), 200)
    with make_client(server) as client:
        response = client.get("https://example.org/")

    assert response.status_code == 200
    assert server.count == 2
    assert len(sleeps) == 1


def test_connect_error_exhausted_raises_last_exception(sleeps: list[float]) -> None:
    errors = [httpx.ConnectError("first"), httpx.ConnectError("last")]
    server = Server(errors[0], errors[1])
    with make_client(server, retry=httpx.RetryPolicy(max_attempts=2)) as client:
        with pytest.raises(httpx.ConnectError) as exc_info:
            client.get("https://example.org/")

    assert exc_info.value is errors[1]
    assert server.count == 2
    assert len(sleeps) == 1


@pytest.mark.parametrize(
    "exc",
    [
        httpx.InvalidURL("invalid"),
        httpx.UnsupportedProtocol("unsupported"),
        httpx.LocalProtocolError("local"),
        httpx.ReadError("read"),
    ],
)
def test_non_retryable_exceptions(sleeps: list[float], exc: Exception) -> None:
    server = Server(exc, 200)
    with make_client(server) as client:
        with pytest.raises(type(exc)):
            client.get("https://example.org/")

    assert server.count == 1
    assert sleeps == []


def test_invalid_url_not_retried(sleeps: list[float]) -> None:
    server = Server(httpx.InvalidURL("invalid"), 200)
    with make_client(server) as client:
        with pytest.raises(httpx.InvalidURL):
            client.get("https://example.org/")

    assert server.count == 1
    assert sleeps == []


def test_unsupported_protocol_not_retried_with_default_transport(
    sleeps: list[float],
) -> None:
    with httpx.Client(retry=httpx.RetryPolicy()) as client:
        with pytest.raises(httpx.UnsupportedProtocol):
            client.get("ftp://example.org/")

    assert sleeps == []


def test_body_read_timeout_is_retried(sleeps: list[float]) -> None:
    class TimeoutStream(httpx.SyncByteStream):
        def __iter__(self) -> typing.Iterator[bytes]:
            raise httpx.ReadTimeout("timed out")

    server = Server(httpx.Response(200, stream=TimeoutStream()), 200)
    with make_client(server) as client:
        response = client.get("https://example.org/")

    assert response.status_code == 200
    assert server.count == 2


# Requests that must not be retried


def test_stream_not_retried(sleeps: list[float]) -> None:
    server = Server(503, 200)
    with make_client(server) as client:
        with client.stream("GET", "https://example.org/") as response:
            assert response.status_code == 503

    assert server.count == 1
    assert sleeps == []


def test_generator_body_not_retried(sleeps: list[float]) -> None:
    def body() -> typing.Iterator[bytes]:
        yield b"data"

    server = Server(503, 200)
    with make_client(server) as client:
        response = client.put("https://example.org/", content=body())

    assert response.status_code == 503
    assert server.count == 1
    assert sleeps == []


def test_iterator_and_file_body_not_retried(sleeps: list[float]) -> None:
    import io

    server = Server(503)
    with make_client(server) as client:
        assert client.put("https://example.org/", content=iter([b"a"])).is_error
        assert client.put("https://example.org/", content=io.BytesIO(b"a")).is_error

    assert server.count == 2


def test_replayable_bodies_are_retried(sleeps: list[float]) -> None:
    import io

    server = Server(503, 200, 503, 200, 503, 200)
    with make_client(server) as client:
        client.put("https://example.org/", content=[b"a", b"b"])
        client.put("https://example.org/", json={"a": 1})
        client.put(
            "https://example.org/",
            files={"file": io.BytesIO(b"content")},
        )

    assert server.count == 6
    assert server.requests[0].content == server.requests[1].content == b"ab"
    assert server.requests[4].content == server.requests[5].content


def test_non_seekable_multipart_not_retried(sleeps: list[float]) -> None:
    class NonSeekable:
        def read(self, size: int = -1) -> bytes:
            return b""

    server = Server(503, 200)
    with make_client(server) as client:
        response = client.put(
            "https://example.org/",
            files={"file": NonSeekable()},  # type: ignore[dict-item]
        )

    assert response.status_code == 503
    assert server.count == 1


def test_custom_stream_body_not_retried(sleeps: list[float]) -> None:
    class Stream(httpx.SyncByteStream):
        def __iter__(self) -> typing.Iterator[bytes]:
            yield b"data"

    server = Server(503, 200)
    with make_client(server) as client:
        request = httpx.Request("PUT", "https://example.org/", stream=Stream())
        response = client.send(request)

    assert response.status_code == 503
    assert server.count == 1


# Response cleanup


def test_previous_responses_closed_before_retry(sleeps: list[float]) -> None:
    closed: list[TrackedStream] = []
    streams = [TrackedStream(closed) for _ in range(3)]
    closed_at_request: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        index = len(closed_at_request)
        closed_at_request.append(len(closed))
        status_code = 503 if index < 2 else 200
        return httpx.Response(status_code, stream=streams[index])

    with make_client(handler) as client:
        response = client.get("https://example.org/")

    assert response.status_code == 200
    assert closed_at_request == [0, 1, 2]
    assert closed == streams


# retry=None


def test_retry_none_matches_default_behavior(sleeps: list[float]) -> None:
    server = Server(503, 200)
    with make_client(server, retry=None) as client:
        response = client.get("https://example.org/")
        assert response.status_code == 503

        server.outcomes = [httpx.ConnectError("failed"), 200]
        with pytest.raises(httpx.ConnectError):
            client.get("https://example.org/")

    assert server.count == 2
    assert sleeps == []


# Interaction with redirects and authentication


def test_each_redirect_hop_has_its_own_budget(sleeps: list[float]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        outcome = outcomes[request.url.path].pop(0)
        if outcome == 302:
            return httpx.Response(302, headers={"Location": "/target"})
        return httpx.Response(outcome)

    paths: list[str] = []
    outcomes = {"/": [503, 503, 302], "/target": [503, 503, 200]}
    with make_client(handler, follow_redirects=True) as client:
        response = client.get("https://example.org/")

    assert response.status_code == 200
    assert paths == ["/", "/", "/", "/target", "/target", "/target"]
    assert len(response.history) == 1
    assert response.history[0].status_code == 302


def test_redirect_to_post_not_retried(sleeps: list[float]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        if request.url.path == "/":
            return httpx.Response(307, headers={"Location": "/target"})
        return httpx.Response(503)

    methods: list[str] = []
    retry = httpx.RetryPolicy(retry_methods={"POST"})
    with make_client(handler, retry=retry, follow_redirects=True) as client:
        client.post("https://example.org/", content=b"a")
    assert methods == ["POST", "POST", "POST", "POST"]

    methods.clear()
    with make_client(handler, follow_redirects=True) as client:
        client.post("https://example.org/", content=b"a")
    assert methods == ["POST", "POST"]


def test_digest_auth_not_counted_as_retries(sleeps: list[float]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        authorization.append(request.headers.get("Authorization"))
        if "Authorization" not in request.headers:
            challenge = 'Digest realm="example", nonce="abc", qop="auth"'
            return httpx.Response(401, headers={"WWW-Authenticate": challenge})
        return httpx.Response(outcomes.pop(0))

    authorization: list[str | None] = []
    outcomes = [503, 200]
    auth = httpx.DigestAuth(username="user", password="pass")
    with make_client(handler, retry=httpx.RetryPolicy(max_attempts=2)) as client:
        response = client.get("https://example.org/", auth=auth)

    assert response.status_code == 200
    assert len(authorization) == 3
    assert authorization[0] is None
    assert authorization[1] is not None and authorization[1].startswith("Digest ")
    assert authorization[2] == authorization[1]
    assert len(response.history) == 1
    assert response.history[0].status_code == 401


def test_basic_auth_retries_keep_credentials(sleeps: list[float]) -> None:
    server = Server(503, 200)
    with make_client(server, auth=("user", "pass")) as client:
        response = client.get("https://example.org/")

    assert response.status_code == 200
    assert [request.headers["Authorization"] for request in server.requests] == [
        "Basic dXNlcjpwYXNz",
        "Basic dXNlcjpwYXNz",
    ]


def test_event_hooks_called_once_per_request(sleeps: list[float]) -> None:
    requests: list[httpx.Request] = []
    responses: list[httpx.Response] = []
    server = Server(503, 200)
    event_hooks = {"request": [requests.append], "response": [responses.append]}
    with make_client(server, event_hooks=event_hooks) as client:
        client.get("https://example.org/")

    assert server.count == 2
    assert len(requests) == 1
    assert [response.status_code for response in responses] == [200]


# AsyncClient


@pytest.mark.anyio
async def test_async_retry_503_then_200(sleeps: list[float]) -> None:
    server = Server(503, 200)
    async with make_async_client(server) as client:
        response = await client.get("https://example.org/")

    assert response.status_code == 200
    assert server.count == 2
    assert len(sleeps) == 1


@pytest.mark.anyio
async def test_async_retry_after(sleeps: list[float]) -> None:
    past = http_date(-datetime.timedelta(hours=1))
    server = Server(
        httpx.Response(429, headers={"Retry-After": "1"}),
        httpx.Response(503, headers={"Retry-After": past}),
        200,
    )
    async with make_async_client(server) as client:
        response = await client.get("https://example.org/")

    assert response.status_code == 200
    assert sleeps == [1.0, 0.0]


@pytest.mark.anyio
async def test_async_post_not_retried(sleeps: list[float]) -> None:
    server = Server(503, 200)
    async with make_async_client(server) as client:
        response = await client.post("https://example.org/", content=b"data")

    assert response.status_code == 503
    assert server.count == 1


@pytest.mark.anyio
async def test_async_exhausts_max_attempts(sleeps: list[float]) -> None:
    server = Server(503)
    async with make_async_client(server) as client:
        response = await client.get("https://example.org/")

    assert response.status_code == 503
    assert server.count == 3
    assert len(sleeps) == 2


@pytest.mark.anyio
async def test_async_connect_error(sleeps: list[float]) -> None:
    server = Server(httpx.ConnectError("failed"), 200)
    async with make_async_client(server) as client:
        response = await client.get("https://example.org/")
    assert response.status_code == 200
    assert server.count == 2

    last = httpx.ConnectError("last")
    server = Server(httpx.ConnectError("first"), last)
    async with make_async_client(
        server, retry=httpx.RetryPolicy(max_attempts=2)
    ) as client:
        with pytest.raises(httpx.ConnectError) as exc_info:
            await client.get("https://example.org/")
    assert exc_info.value is last


@pytest.mark.anyio
async def test_async_invalid_url_not_retried(sleeps: list[float]) -> None:
    server = Server(httpx.InvalidURL("invalid"), 200)
    async with make_async_client(server) as client:
        with pytest.raises(httpx.InvalidURL):
            await client.get("https://example.org/")

    assert server.count == 1
    assert sleeps == []


@pytest.mark.anyio
async def test_async_body_read_timeout_is_retried(sleeps: list[float]) -> None:
    class TimeoutStream(httpx.AsyncByteStream):
        async def __aiter__(self) -> typing.AsyncIterator[bytes]:
            raise httpx.ReadTimeout("timed out")
            yield b""  # pragma: no cover

    server = Server(httpx.Response(200, stream=TimeoutStream()), 200)
    async with make_async_client(server) as client:
        response = await client.get("https://example.org/")

    assert response.status_code == 200
    assert server.count == 2


@pytest.mark.anyio
async def test_async_stream_not_retried(sleeps: list[float]) -> None:
    server = Server(503, 200)
    async with make_async_client(server) as client:
        async with client.stream("GET", "https://example.org/") as response:
            assert response.status_code == 503

    assert server.count == 1


@pytest.mark.anyio
async def test_async_generator_body_not_retried(sleeps: list[float]) -> None:
    async def body() -> typing.AsyncIterator[bytes]:
        yield b"data"

    def handler(request: httpx.Request) -> httpx.Response:
        count.append(request)
        return httpx.Response(503)

    count: list[httpx.Request] = []
    async with make_async_client(handler) as client:
        response = await client.put("https://example.org/", content=body())

    assert response.status_code == 503
    assert len(count) == 1


@pytest.mark.anyio
async def test_async_previous_responses_closed_before_retry(
    sleeps: list[float],
) -> None:
    closed: list[TrackedStream] = []
    streams = [TrackedStream(closed) for _ in range(3)]
    closed_at_request: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        index = len(closed_at_request)
        closed_at_request.append(len(closed))
        status_code = 503 if index < 2 else 200
        return httpx.Response(status_code, stream=streams[index])

    async with make_async_client(handler) as client:
        response = await client.get("https://example.org/")

    assert response.status_code == 200
    assert closed_at_request == [0, 1, 2]
    assert closed == streams


@pytest.mark.anyio
async def test_async_redirect_hops_have_own_budget(sleeps: list[float]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        outcome = outcomes[request.url.path].pop(0)
        if outcome == 302:
            return httpx.Response(302, headers={"Location": "/target"})
        return httpx.Response(outcome)

    paths: list[str] = []
    outcomes = {"/": [503, 302], "/target": [503, 200]}
    async with make_async_client(handler, follow_redirects=True) as client:
        response = await client.get("https://example.org/")

    assert response.status_code == 200
    assert paths == ["/", "/", "/target", "/target"]


@pytest.mark.anyio
async def test_async_retry_none(sleeps: list[float]) -> None:
    server = Server(503, 200)
    async with make_async_client(server, retry=None) as client:
        response = await client.get("https://example.org/")

    assert response.status_code == 503
    assert server.count == 1
    assert sleeps == []
