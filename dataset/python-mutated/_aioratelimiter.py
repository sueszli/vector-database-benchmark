"""This module contains an implementation of the BaseRateLimiter class based on the aiolimiter
library.
"""
import asyncio
import contextlib
import sys
from typing import Any, AsyncIterator, Callable, Coroutine, Dict, List, Optional, Union
try:
    from aiolimiter import AsyncLimiter
    AIO_LIMITER_AVAILABLE = True
except ImportError:
    AIO_LIMITER_AVAILABLE = False
from telegram._utils.logging import get_logger
from telegram._utils.types import JSONDict
from telegram.error import RetryAfter
from telegram.ext._baseratelimiter import BaseRateLimiter
if sys.version_info >= (3, 10):
    null_context = contextlib.nullcontext
else:

    @contextlib.asynccontextmanager
    async def null_context() -> AsyncIterator[None]:
        yield None
_LOGGER = get_logger(__name__, class_name='AIORateLimiter')

class AIORateLimiter(BaseRateLimiter[int]):
    """
    Implementation of :class:`~telegram.ext.BaseRateLimiter` using the library
    `aiolimiter <https://aiolimiter.readthedocs.io/en/stable>`_.

    Important:
        If you want to use this class, you must install PTB with the optional requirement
        ``rate-limiter``, i.e.

        .. code-block:: bash

           pip install "python-telegram-bot[rate-limiter]"

    The rate limiting is applied by combining two levels of throttling and :meth:`process_request`
    roughly boils down to::

        async with group_limiter(group_id):
            async with overall_limiter:
                await callback(*args, **kwargs)

    Here, ``group_id`` is determined by checking if there is a ``chat_id`` parameter in the
    :paramref:`~telegram.ext.BaseRateLimiter.process_request.data`.
    The ``overall_limiter`` is applied only if a ``chat_id`` argument is present at all.

    Attention:
        * Some bot methods accept a ``chat_id`` parameter in form of a ``@username`` for
          supergroups and channels. As we can't know which ``@username`` corresponds to which
          integer ``chat_id``, these will be treated as different groups, which may lead to
          exceeding the rate limit.
        * As channels can't be differentiated from supergroups by the ``@username`` or integer
          ``chat_id``, this also applies the group related rate limits to channels.
        * A :exc:`~telegram.error.RetryAfter` exception will halt *all* requests for
          :attr:`~telegram.error.RetryAfter.retry_after` + 0.1 seconds. This may be stricter than
          necessary in some cases, e.g. the bot may hit a rate limit in one group but might still
          be allowed to send messages in another group.

    Note:
        This class is to be understood as minimal effort reference implementation.
        If you would like to handle rate limiting in a more sophisticated, fine-tuned way, we
        welcome you to implement your own subclass of :class:`~telegram.ext.BaseRateLimiter`.
        Feel free to check out the source code of this class for inspiration.

    .. seealso:: :wiki:`Avoiding Flood Limits <Avoiding-flood-limits>`

    .. versionadded:: 20.0

    Args:
        overall_max_rate (:obj:`float`): The maximum number of requests allowed for the entire bot
            per :paramref:`overall_time_period`. When set to 0, no rate limiting will be applied.
            Defaults to ``30``.
        overall_time_period (:obj:`float`): The time period (in seconds) during which the
            :paramref:`overall_max_rate` is enforced.  When set to 0, no rate limiting will be
            applied. Defaults to 1.
        group_max_rate (:obj:`float`): The maximum number of requests allowed for requests related
            to groups and channels per :paramref:`group_time_period`.  When set to 0, no rate
            limiting will be applied. Defaults to 20.
        group_time_period (:obj:`float`): The time period (in seconds) during which the
            :paramref:`group_max_rate` is enforced.  When set to 0, no rate limiting will be
            applied. Defaults to 60.
        max_retries (:obj:`int`): The maximum number of retries to be made in case of a
            :exc:`~telegram.error.RetryAfter` exception.
            If set to 0, no retries will be made. Defaults to ``0``.

    """
    __slots__ = ('_base_limiter', '_group_limiters', '_group_max_rate', '_group_time_period', '_max_retries', '_retry_after_event')

    def __init__(self, overall_max_rate: float=30, overall_time_period: float=1, group_max_rate: float=20, group_time_period: float=60, max_retries: int=0) -> None:
        if False:
            i = 10
            return i + 15
        if not AIO_LIMITER_AVAILABLE:
            raise RuntimeError('To use `AIORateLimiter`, PTB must be installed via `pip install "python-telegram-bot[rate-limiter]"`.')
        if overall_max_rate and overall_time_period:
            self._base_limiter: Optional[AsyncLimiter] = AsyncLimiter(max_rate=overall_max_rate, time_period=overall_time_period)
        else:
            self._base_limiter = None
        if group_max_rate and group_time_period:
            self._group_max_rate: float = group_max_rate
            self._group_time_period: float = group_time_period
        else:
            self._group_max_rate = 0
            self._group_time_period = 0
        self._group_limiters: Dict[Union[str, int], AsyncLimiter] = {}
        self._max_retries: int = max_retries
        self._retry_after_event = asyncio.Event()
        self._retry_after_event.set()

    async def initialize(self) -> None:
        """Does nothing."""

    async def shutdown(self) -> None:
        """Does nothing."""

    def _get_group_limiter(self, group_id: Union[str, int, bool]) -> 'AsyncLimiter':
        if False:
            i = 10
            return i + 15
        if len(self._group_limiters) > 512:
            for (key, limiter) in self._group_limiters.copy().items():
                if key == group_id:
                    continue
                if limiter.has_capacity(limiter.max_rate):
                    del self._group_limiters[key]
        if group_id not in self._group_limiters:
            self._group_limiters[group_id] = AsyncLimiter(max_rate=self._group_max_rate, time_period=self._group_time_period)
        return self._group_limiters[group_id]

    async def _run_request(self, chat: bool, group: Union[str, int, bool], callback: Callable[..., Coroutine[Any, Any, Union[bool, JSONDict, List[JSONDict]]]], args: Any, kwargs: Dict[str, Any]) -> Union[bool, JSONDict, List[JSONDict]]:
        base_context = self._base_limiter if chat and self._base_limiter else null_context()
        group_context = self._get_group_limiter(group) if group and self._group_max_rate else null_context()
        async with group_context, base_context:
            await self._retry_after_event.wait()
            return await callback(*args, **kwargs)

    async def process_request(self, callback: Callable[..., Coroutine[Any, Any, Union[bool, JSONDict, List[JSONDict]]]], args: Any, kwargs: Dict[str, Any], endpoint: str, data: Dict[str, Any], rate_limit_args: Optional[int]) -> Union[bool, JSONDict, List[JSONDict]]:
        """
        Processes a request by applying rate limiting.

        See :meth:`telegram.ext.BaseRateLimiter.process_request` for detailed information on the
        arguments.

        Args:
            rate_limit_args (:obj:`None` | :obj:`int`): If set, specifies the maximum number of
                retries to be made in case of a :exc:`~telegram.error.RetryAfter` exception.
                Defaults to :paramref:`AIORateLimiter.max_retries`.
        """
        max_retries = rate_limit_args or self._max_retries
        group: Union[int, str, bool] = False
        chat: bool = False
        chat_id = data.get('chat_id')
        if chat_id is not None:
            chat = True
        with contextlib.suppress(ValueError, TypeError):
            chat_id = int(chat_id)
        if isinstance(chat_id, int) and chat_id < 0 or isinstance(chat_id, str):
            group = chat_id
        for i in range(max_retries + 1):
            try:
                return await self._run_request(chat=chat, group=group, callback=callback, args=args, kwargs=kwargs)
            except RetryAfter as exc:
                if i == max_retries:
                    _LOGGER.exception('Rate limit hit after maximum of %d retries', max_retries, exc_info=exc)
                    raise exc
                sleep = exc.retry_after + 0.1
                _LOGGER.info('Rate limit hit. Retrying after %f seconds', sleep)
                self._retry_after_event.clear()
                await asyncio.sleep(sleep)
            finally:
                self._retry_after_event.set()
        return None