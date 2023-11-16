import asyncio
from ._http_response_impl_async import AsyncHttpResponseImpl
from ._requests_basic import _RestRequestsTransportResponseBase
from ..pipeline.transport._requests_asyncio import AsyncioStreamDownloadGenerator

class RestAsyncioRequestsTransportResponse(AsyncHttpResponseImpl, _RestRequestsTransportResponseBase):
    """Asynchronous streaming of data from the response."""

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(stream_download_generator=AsyncioStreamDownloadGenerator, **kwargs)

    async def close(self) -> None:
        """Close the response.

        :return: None
        :rtype: None
        """
        if not self.is_closed:
            self._is_closed = True
            self._internal_response.close()
            await asyncio.sleep(0)