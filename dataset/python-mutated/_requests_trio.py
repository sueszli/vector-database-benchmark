import trio
from ._http_response_impl_async import AsyncHttpResponseImpl
from ._requests_basic import _RestRequestsTransportResponseBase
from ..pipeline.transport._requests_trio import TrioStreamDownloadGenerator

class RestTrioRequestsTransportResponse(AsyncHttpResponseImpl, _RestRequestsTransportResponseBase):
    """Asynchronous streaming of data from the response."""

    def __init__(self, **kwargs):
        if False:
            return 10
        super().__init__(stream_download_generator=TrioStreamDownloadGenerator, **kwargs)

    async def close(self) -> None:
        if not self.is_closed:
            self._is_closed = True
            self._internal_response.close()
            await trio.sleep(0)