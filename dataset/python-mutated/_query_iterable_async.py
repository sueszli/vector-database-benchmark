"""Iterable query results in the Azure Cosmos database service.
"""
import asyncio
from azure.core.async_paging import AsyncPageIterator
from azure.cosmos._execution_context.aio import execution_dispatcher

class QueryIterable(AsyncPageIterator):
    """Represents an iterable object of the query results.

    QueryIterable is a wrapper for query execution context.
    """

    def __init__(self, client, query, options, fetch_function=None, collection_link=None, database_link=None, partition_key=None, continuation_token=None):
        if False:
            return 10
        "Instantiates a QueryIterable for non-client side partitioning queries.\n\n        _ProxyQueryExecutionContext will be used as the internal query execution\n        context.\n\n        :param CosmosClient client: Instance of document client.\n        :param (str or dict) query:\n        :param dict options: The request options for the request.\n        :param method fetch_function:\n        :param method resource_type: The type of the resource being queried\n        :param str resource_link: If this is a Document query/feed collection_link is required.\n\n        Example of `fetch_function`:\n\n        >>> def result_fn(result):\n        >>>     return result['Databases']\n\n        "
        self._client = client
        self.retry_options = client.connection_policy.RetryOptions
        self._query = query
        self._options = options
        if continuation_token:
            options['continuation'] = continuation_token
        self._fetch_function = fetch_function
        self._collection_link = collection_link
        self._database_link = database_link
        self._partition_key = partition_key
        self._ex_context = execution_dispatcher._ProxyQueryExecutionContext(self._client, self._collection_link, self._query, self._options, self._fetch_function)
        super(QueryIterable, self).__init__(self._fetch_next, self._unpack, continuation_token=continuation_token)

    async def _unpack(self, block):
        continuation = None
        if self._client.last_response_headers:
            continuation = self._client.last_response_headers.get('x-ms-continuation') or self._client.last_response_headers.get('etag')
        if block:
            self._did_a_call_already = False
        return (continuation, block)

    async def _fetch_next(self, *args):
        """Return a block of results with respecting retry policy.

        This method only exists for backward compatibility reasons. (Because
        QueryIterable has exposed fetch_next_block api).

        :param Any args:
        :return: List of results.
        :rtype: list
        """
        if 'partitionKey' in self._options and asyncio.iscoroutine(self._options['partitionKey']):
            self._options['partitionKey'] = await self._options['partitionKey']
        block = await self._ex_context.fetch_next_block()
        if not block:
            raise StopAsyncIteration
        return block