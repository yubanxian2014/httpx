from __future__ import annotations

import datetime
import email.utils
import random
import typing

from ._content import AsyncIteratorByteStream, ByteStream, IteratorByteStream
from ._exceptions import (
    ConnectError,
    ConnectTimeout,
    PoolTimeout,
    ReadTimeout,
    RemoteProtocolError,
    WriteTimeout,
)
from ._models import Request, Response
from ._multipart import FileField, MultipartStream

__all__ = ["RetryPolicy"]

DEFAULT_RETRY_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})
DEFAULT_RETRY_METHODS: frozenset[str] = frozenset(
    {"GET", "HEAD", "PUT", "DELETE", "OPTIONS", "TRACE"}
)

RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    ConnectError,
    ConnectTimeout,
    ReadTimeout,
    WriteTimeout,
    PoolTimeout,
    RemoteProtocolError,
)

# Cap the exponent so that very large attempt counts can't overflow a float.
_MAX_BACKOFF_EXPONENT = 64


class RetryPolicy:
    """
    Configuration for automatically retrying failed requests.

    Retries apply to each individual network request sent by the client.
    Every hop of a redirect chain, and every request made by an authentication
    flow, gets its own independent budget of `max_attempts`.

    Requests are only retried when the method is in `retry_methods`, the
    request body can be safely replayed, and the response was not requested
    with `stream=True`.

    **Parameters:**

    * **max_attempts** - The total number of attempts, including the first one.
    * **backoff_factor** - The base delay in seconds. The delay before the n-th
            retry is `min(max_backoff, backoff_factor * 2 ** (n - 1))`, with full
            jitter applied.
    * **max_backoff** - The maximum delay in seconds between attempts. Also caps
            any `Retry-After` delay.
    * **retry_status_codes** - Response status codes that should be retried.
    * **retry_methods** - Request methods that may be retried.
    * **respect_retry_after** - Use the `Retry-After` response header, when
            present, instead of the backoff formula.

    Usage:

    ```python
    >>> retry = httpx.RetryPolicy(max_attempts=5, backoff_factor=1.0)
    >>> client = httpx.Client(retry=retry)
    ```
    """

    def __init__(
        self,
        *,
        max_attempts: int = 3,
        backoff_factor: float = 0.5,
        max_backoff: float = 60.0,
        retry_status_codes: typing.Collection[int] = DEFAULT_RETRY_STATUS_CODES,
        retry_methods: typing.Collection[str] = DEFAULT_RETRY_METHODS,
        respect_retry_after: bool = True,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1.")
        if backoff_factor < 0:
            raise ValueError("backoff_factor must not be negative.")
        if max_backoff < 0:
            raise ValueError("max_backoff must not be negative.")

        self.max_attempts = max_attempts
        self.backoff_factor = backoff_factor
        self.max_backoff = max_backoff
        self.retry_status_codes = frozenset(retry_status_codes)
        self.retry_methods = frozenset(method.upper() for method in retry_methods)
        self.respect_retry_after = respect_retry_after

    def is_retryable_request(self, request: Request) -> bool:
        """
        Return `True` if the request may be retried, based on its method and
        on whether its body can be replayed.
        """
        return request.method.upper() in self.retry_methods and _is_replayable(request)

    def is_retryable_exception(self, exc: Exception) -> bool:
        """
        Return `True` if a request that raised `exc` may be retried.
        """
        return isinstance(exc, RETRYABLE_EXCEPTIONS)

    def is_retryable_response(self, response: Response) -> bool:
        """
        Return `True` if the response status code should be retried.
        """
        return response.status_code in self.retry_status_codes

    def get_backoff(self, retry_number: int) -> float:
        """
        Return the jittered backoff delay in seconds before the given retry,
        where `retry_number` starts at 1 for the first retry.
        """
        exponent = min(retry_number - 1, _MAX_BACKOFF_EXPONENT)
        delay = min(self.max_backoff, self.backoff_factor * 2**exponent)
        return random.uniform(0, delay)

    def get_retry_after(self, response: Response) -> float | None:
        """
        Return the delay in seconds requested by the response's `Retry-After`
        header, capped at `max_backoff`, or `None` if there is no valid header.
        """
        value = response.headers.get("Retry-After")
        if value is None:
            return None

        value = value.strip()
        if value.isascii() and value.isdigit():
            seconds = float(value)
        else:
            try:
                retry_at = email.utils.parsedate_to_datetime(value)
            except (TypeError, ValueError, IndexError, OverflowError):
                return None
            if retry_at is None:  # pragma: no cover
                return None
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=datetime.timezone.utc)
            now = datetime.datetime.now(datetime.timezone.utc)
            seconds = (retry_at - now).total_seconds()

        return min(self.max_backoff, max(0.0, seconds))

    def get_delay(self, retry_number: int, response: Response | None = None) -> float:
        """
        Return the delay in seconds before the given retry. `Retry-After` takes
        precedence over the backoff formula when `respect_retry_after` is set.
        """
        if response is not None and self.respect_retry_after:
            retry_after = self.get_retry_after(response)
            if retry_after is not None:
                return retry_after
        return self.get_backoff(retry_number)

    def __eq__(self, other: typing.Any) -> bool:
        return (
            isinstance(other, self.__class__)
            and self.max_attempts == other.max_attempts
            and self.backoff_factor == other.backoff_factor
            and self.max_backoff == other.max_backoff
            and self.retry_status_codes == other.retry_status_codes
            and self.retry_methods == other.retry_methods
            and self.respect_retry_after == other.respect_retry_after
        )

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return (
            f"{class_name}(max_attempts={self.max_attempts}, "
            f"backoff_factor={self.backoff_factor}, "
            f"max_backoff={self.max_backoff}, "
            f"retry_status_codes={sorted(self.retry_status_codes)}, "
            f"retry_methods={sorted(self.retry_methods)}, "
            f"respect_retry_after={self.respect_retry_after})"
        )


def _is_replayable(request: Request) -> bool:
    """
    Return `True` if the request body can be sent more than once.
    """
    stream = request.stream
    if isinstance(stream, ByteStream):
        return True
    if isinstance(stream, MultipartStream):
        return all(
            isinstance(field.file, (str, bytes))
            or (hasattr(field.file, "seekable") and field.file.seekable())
            for field in stream.fields
            if isinstance(field, FileField)
        )
    if isinstance(stream, IteratorByteStream):
        content = stream._stream
        return not (hasattr(content, "read") or isinstance(content, typing.Iterator))
    if isinstance(stream, AsyncIteratorByteStream):
        async_content = stream._stream
        return not (
            hasattr(async_content, "aread")
            or isinstance(async_content, typing.AsyncIterator)
        )
    return False
