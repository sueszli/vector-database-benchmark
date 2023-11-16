import collections.abc
import logging
from typing import Iterable, AsyncIterator, TypeVar, Callable, Tuple, Optional, Awaitable, Any
from .exceptions import AzureError
_LOGGER = logging.getLogger(__name__)
ReturnType = TypeVar('ReturnType')
ResponseType = TypeVar('ResponseType')
__all__ = ['AsyncPageIterator', 'AsyncItemPaged']

class AsyncList(AsyncIterator[ReturnType]):

    def __init__(self, iterable: Iterable[ReturnType]) -> None:
        if False:
            i = 10
            return i + 15
        'Change an iterable into a fake async iterator.\n\n        Could be useful to fill the async iterator contract when you get a list.\n\n        :param iterable: A sync iterable of T\n        '
        self._iterator = iter(iterable)

    async def __anext__(self) -> ReturnType:
        try:
            return next(self._iterator)
        except StopIteration as err:
            raise StopAsyncIteration() from err

class AsyncPageIterator(AsyncIterator[AsyncIterator[ReturnType]]):

    def __init__(self, get_next: Callable[[Optional[str]], Awaitable[ResponseType]], extract_data: Callable[[ResponseType], Awaitable[Tuple[str, AsyncIterator[ReturnType]]]], continuation_token: Optional[str]=None) -> None:
        if False:
            print('Hello World!')
        'Return an async iterator of pages.\n\n        :param get_next: Callable that take the continuation token and return a HTTP response\n        :param extract_data: Callable that take an HTTP response and return a tuple continuation token,\n         list of ReturnType\n        :param str continuation_token: The continuation token needed by get_next\n        '
        self._get_next = get_next
        self._extract_data = extract_data
        self.continuation_token = continuation_token
        self._did_a_call_already = False
        self._response: Optional[ResponseType] = None
        self._current_page: Optional[AsyncIterator[ReturnType]] = None

    async def __anext__(self) -> AsyncIterator[ReturnType]:
        if self.continuation_token is None and self._did_a_call_already:
            raise StopAsyncIteration('End of paging')
        try:
            self._response = await self._get_next(self.continuation_token)
        except AzureError as error:
            if not error.continuation_token:
                error.continuation_token = self.continuation_token
            raise
        self._did_a_call_already = True
        (self.continuation_token, self._current_page) = await self._extract_data(self._response)
        if isinstance(self._current_page, collections.abc.Iterable):
            self._current_page = AsyncList(self._current_page)
        return self._current_page

class AsyncItemPaged(AsyncIterator[ReturnType]):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        'Return an async iterator of items.\n\n        args and kwargs will be passed to the AsyncPageIterator constructor directly,\n        except page_iterator_class\n        '
        self._args = args
        self._kwargs = kwargs
        self._page_iterator: Optional[AsyncIterator[AsyncIterator[ReturnType]]] = None
        self._page: Optional[AsyncIterator[ReturnType]] = None
        self._page_iterator_class = self._kwargs.pop('page_iterator_class', AsyncPageIterator)

    def by_page(self, continuation_token: Optional[str]=None) -> AsyncIterator[AsyncIterator[ReturnType]]:
        if False:
            while True:
                i = 10
        'Get an async iterator of pages of objects, instead of an async iterator of objects.\n\n        :param str continuation_token:\n            An opaque continuation token. This value can be retrieved from the\n            continuation_token field of a previous generator object. If specified,\n            this generator will begin returning results from this point.\n        :returns: An async iterator of pages (themselves async iterator of objects)\n        :rtype: AsyncIterator[AsyncIterator[ReturnType]]\n        '
        return self._page_iterator_class(*self._args, **self._kwargs, continuation_token=continuation_token)

    async def __anext__(self) -> ReturnType:
        if self._page_iterator is None:
            self._page_iterator = self.by_page()
            return await self.__anext__()
        if self._page is None:
            self._page = await self._page_iterator.__anext__()
            return await self.__anext__()
        try:
            return await self._page.__anext__()
        except StopAsyncIteration:
            self._page = None
            return await self.__anext__()