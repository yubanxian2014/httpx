HTTPX can optionally retry failed requests. Retries are disabled by default,
and are enabled by passing a `RetryPolicy` to the client:

```python
retry = httpx.RetryPolicy(max_attempts=5, backoff_factor=1.0)

with httpx.Client(retry=retry) as client:
    response = client.get("https://www.example.com/")
```

The same configuration is supported by `httpx.AsyncClient`.

## What gets retried

A request is retried when its method is in `retry_methods` and either:

* The response status code is in `retry_status_codes`. When the attempts are
exhausted the last response is returned, rather than raising an exception.
* One of `ConnectError`, `ConnectTimeout`, `ReadTimeout`, `WriteTimeout`,
`PoolTimeout` or `RemoteProtocolError` is raised. When the attempts are exhausted
the last exception is raised.

By default `POST` and `PATCH` requests are not retried.

Requests are never retried when using `stream=True` / `client.stream(...)`, or
when the request body cannot be replayed, such as a generator, an iterator,
or a non-seekable file.

## Delays between attempts

The delay before the n-th retry is
`min(max_backoff, backoff_factor * 2 ** (n - 1))`, with full jitter applied.

If `respect_retry_after` is enabled and the response includes a `Retry-After`
header, either as a number of seconds or as an HTTP date, then that delay is used
instead, capped at `max_backoff`.

## Redirects and authentication

Retries apply to each individual network request. Each hop of a redirect chain,
and each request made by an authentication flow, has its own budget of
`max_attempts`. Event hooks are called once per request, rather than once per
attempt.
