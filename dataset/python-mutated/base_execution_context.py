"""Internal class for query execution context implementation in the Azure Cosmos
database service.
"""
from collections import deque
import copy
from ...aio import _retry_utility_async
from ... import http_constants

class _QueryExecutionContextBase(object):
    """
    This is the abstract base execution context class.
    """

    def __init__(self, client, options):
        if False:
            while True:
                i = 10
        '\n        :param CosmosClient client:\n        :param dict options: The request options for the request.\n        '
        self._client = client
        self._options = options
        self._is_change_feed = 'changeFeed' in options and options['changeFeed'] is True
        self._continuation = self._get_initial_continuation()
        self._has_started = False
        self._has_finished = False
        self._buffer = deque()

    def _get_initial_continuation(self):
        if False:
            return 10
        if 'continuation' in self._options:
            return self._options['continuation']
        return None

    def _has_more_pages(self):
        if False:
            while True:
                i = 10
        return not self._has_started or self._continuation

    async def fetch_next_block(self):
        """Returns a block of results with respecting retry policy.

        This method only exists for backward compatibility reasons. (Because
        QueryIterable has exposed fetch_next_block api).

        :return: List of results.
        :rtype: list
        """
        if not self._has_more_pages():
            return []
        if self._buffer:
            res = list(self._buffer)
            self._buffer.clear()
            return res
        return await self._fetch_next_block()

    async def _fetch_next_block(self):
        raise NotImplementedError

    def __aiter__(self):
        if False:
            print('Hello World!')
        'Returns itself as an iterator\n        :returns: Query as an iterator.\n        :rtype: Iterator\n        '
        return self

    async def __anext__(self):
        """Return the next query result.

        :return: The next query result.
        :rtype: dict
        :raises StopAsyncIteration: If no more result is left.
        """
        if self._has_finished:
            raise StopAsyncIteration
        if not self._buffer:
            results = await self.fetch_next_block()
            self._buffer.extend(results)
        if not self._buffer:
            raise StopAsyncIteration
        return self._buffer.popleft()

    async def _fetch_items_helper_no_retries(self, fetch_function):
        """Fetches more items and doesn't retry on failure

        :param Callable fetch_function: The function that fetches the items.
        :return: List of fetched items.
        :rtype: list
        """
        fetched_items = []
        while self._continuation or not self._has_started:
            if not self._has_started:
                self._has_started = True
            new_options = copy.deepcopy(self._options)
            new_options['continuation'] = self._continuation
            (fetched_items, response_headers) = await fetch_function(new_options)
            continuation_key = http_constants.HttpHeaders.Continuation
            if self._is_change_feed:
                continuation_key = http_constants.HttpHeaders.ETag
            if not self._is_change_feed or fetched_items:
                self._continuation = response_headers.get(continuation_key)
            else:
                self._continuation = None
            if fetched_items:
                break
        return fetched_items

    async def _fetch_items_helper_with_retries(self, fetch_function):

        async def callback():
            return await self._fetch_items_helper_no_retries(fetch_function)
        return await _retry_utility_async.ExecuteAsync(self._client, self._client._global_endpoint_manager, callback)

class _DefaultQueryExecutionContext(_QueryExecutionContextBase):
    """
    This is the default execution context.
    """

    def __init__(self, client, options, fetch_function):
        if False:
            return 10
        "\n        :param CosmosClient client:\n        :param dict options: The request options for the request.\n        :param method fetch_function:\n            Will be invoked for retrieving each page\n\n            Example of `fetch_function`:\n\n            >>> def result_fn(result):\n            >>>     return result['Databases']\n\n        "
        super(_DefaultQueryExecutionContext, self).__init__(client, options)
        self._fetch_function = fetch_function

    async def _fetch_next_block(self):
        while super(_DefaultQueryExecutionContext, self)._has_more_pages() and (not self._buffer):
            return await self._fetch_items_helper_with_retries(self._fetch_function)