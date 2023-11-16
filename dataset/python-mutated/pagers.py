from typing import Any, AsyncIterator, Awaitable, Callable, Iterator, Optional, Sequence, Tuple
from google.cloud.container_v1beta1.types import cluster_service

class ListUsableSubnetworksPager:
    """A pager for iterating through ``list_usable_subnetworks`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.container_v1beta1.types.ListUsableSubnetworksResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``subnetworks`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListUsableSubnetworks`` requests and continue to iterate
    through the ``subnetworks`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.container_v1beta1.types.ListUsableSubnetworksResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(self, method: Callable[..., cluster_service.ListUsableSubnetworksResponse], request: cluster_service.ListUsableSubnetworksRequest, response: cluster_service.ListUsableSubnetworksResponse, *, metadata: Sequence[Tuple[str, str]]=()):
        if False:
            return 10
        'Instantiate the pager.\n\n        Args:\n            method (Callable): The method that was originally called, and\n                which instantiated this pager.\n            request (google.cloud.container_v1beta1.types.ListUsableSubnetworksRequest):\n                The initial request object.\n            response (google.cloud.container_v1beta1.types.ListUsableSubnetworksResponse):\n                The initial response object.\n            metadata (Sequence[Tuple[str, str]]): Strings which should be\n                sent along with the request as metadata.\n        '
        self._method = method
        self._request = cluster_service.ListUsableSubnetworksRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        if False:
            for i in range(10):
                print('nop')
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[cluster_service.ListUsableSubnetworksResponse]:
        if False:
            return 10
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(self._request, metadata=self._metadata)
            yield self._response

    def __iter__(self) -> Iterator[cluster_service.UsableSubnetwork]:
        if False:
            while True:
                i = 10
        for page in self.pages:
            yield from page.subnetworks

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return '{0}<{1!r}>'.format(self.__class__.__name__, self._response)

class ListUsableSubnetworksAsyncPager:
    """A pager for iterating through ``list_usable_subnetworks`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.container_v1beta1.types.ListUsableSubnetworksResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``subnetworks`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListUsableSubnetworks`` requests and continue to iterate
    through the ``subnetworks`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.container_v1beta1.types.ListUsableSubnetworksResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(self, method: Callable[..., Awaitable[cluster_service.ListUsableSubnetworksResponse]], request: cluster_service.ListUsableSubnetworksRequest, response: cluster_service.ListUsableSubnetworksResponse, *, metadata: Sequence[Tuple[str, str]]=()):
        if False:
            print('Hello World!')
        'Instantiates the pager.\n\n        Args:\n            method (Callable): The method that was originally called, and\n                which instantiated this pager.\n            request (google.cloud.container_v1beta1.types.ListUsableSubnetworksRequest):\n                The initial request object.\n            response (google.cloud.container_v1beta1.types.ListUsableSubnetworksResponse):\n                The initial response object.\n            metadata (Sequence[Tuple[str, str]]): Strings which should be\n                sent along with the request as metadata.\n        '
        self._method = method
        self._request = cluster_service.ListUsableSubnetworksRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        if False:
            print('Hello World!')
        return getattr(self._response, name)

    @property
    async def pages(self) -> AsyncIterator[cluster_service.ListUsableSubnetworksResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = await self._method(self._request, metadata=self._metadata)
            yield self._response

    def __aiter__(self) -> AsyncIterator[cluster_service.UsableSubnetwork]:
        if False:
            i = 10
            return i + 15

        async def async_generator():
            async for page in self.pages:
                for response in page.subnetworks:
                    yield response
        return async_generator()

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return '{0}<{1!r}>'.format(self.__class__.__name__, self._response)