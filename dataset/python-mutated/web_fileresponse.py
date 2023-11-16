import asyncio
import mimetypes
import os
import pathlib
from typing import IO, TYPE_CHECKING, Any, Awaitable, Callable, Final, Optional, Tuple, cast
from . import hdrs
from .abc import AbstractStreamWriter
from .helpers import ETAG_ANY, ETag, must_be_empty_body
from .typedefs import LooseHeaders, PathLike
from .web_exceptions import HTTPNotModified, HTTPPartialContent, HTTPPreconditionFailed, HTTPRequestRangeNotSatisfiable
from .web_response import StreamResponse
__all__ = ('FileResponse',)
if TYPE_CHECKING:
    from .web_request import BaseRequest
_T_OnChunkSent = Optional[Callable[[bytes], Awaitable[None]]]
NOSENDFILE: Final[bool] = bool(os.environ.get('AIOHTTP_NOSENDFILE'))

class FileResponse(StreamResponse):
    """A response object can be used to send files."""

    def __init__(self, path: PathLike, chunk_size: int=256 * 1024, status: int=200, reason: Optional[str]=None, headers: Optional[LooseHeaders]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(status=status, reason=reason, headers=headers)
        self._path = pathlib.Path(path)
        self._chunk_size = chunk_size

    async def _sendfile_fallback(self, writer: AbstractStreamWriter, fobj: IO[Any], offset: int, count: int) -> AbstractStreamWriter:
        chunk_size = self._chunk_size
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, fobj.seek, offset)
        chunk = await loop.run_in_executor(None, fobj.read, chunk_size)
        while chunk:
            await writer.write(chunk)
            count = count - chunk_size
            if count <= 0:
                break
            chunk = await loop.run_in_executor(None, fobj.read, min(chunk_size, count))
        await writer.drain()
        return writer

    async def _sendfile(self, request: 'BaseRequest', fobj: IO[Any], offset: int, count: int) -> AbstractStreamWriter:
        writer = await super().prepare(request)
        assert writer is not None
        if NOSENDFILE or self.compression:
            return await self._sendfile_fallback(writer, fobj, offset, count)
        loop = request._loop
        transport = request.transport
        assert transport is not None
        try:
            await loop.sendfile(transport, fobj, offset, count)
        except NotImplementedError:
            return await self._sendfile_fallback(writer, fobj, offset, count)
        await super().write_eof()
        return writer

    @staticmethod
    def _strong_etag_match(etag_value: str, etags: Tuple[ETag, ...]) -> bool:
        if False:
            print('Hello World!')
        if len(etags) == 1 and etags[0].value == ETAG_ANY:
            return True
        return any((etag.value == etag_value for etag in etags if not etag.is_weak))

    async def _not_modified(self, request: 'BaseRequest', etag_value: str, last_modified: float) -> Optional[AbstractStreamWriter]:
        self.set_status(HTTPNotModified.status_code)
        self._length_check = False
        self.etag = etag_value
        self.last_modified = last_modified
        return await super().prepare(request)

    async def _precondition_failed(self, request: 'BaseRequest') -> Optional[AbstractStreamWriter]:
        self.set_status(HTTPPreconditionFailed.status_code)
        self.content_length = 0
        return await super().prepare(request)

    async def prepare(self, request: 'BaseRequest') -> Optional[AbstractStreamWriter]:
        filepath = self._path
        gzip = False
        if 'gzip' in request.headers.get(hdrs.ACCEPT_ENCODING, ''):
            gzip_path = filepath.with_name(filepath.name + '.gz')
            if gzip_path.is_file():
                filepath = gzip_path
                gzip = True
        loop = asyncio.get_event_loop()
        st: os.stat_result = await loop.run_in_executor(None, filepath.stat)
        etag_value = f'{st.st_mtime_ns:x}-{st.st_size:x}'
        last_modified = st.st_mtime
        ifmatch = request.if_match
        if ifmatch is not None and (not self._strong_etag_match(etag_value, ifmatch)):
            return await self._precondition_failed(request)
        unmodsince = request.if_unmodified_since
        if unmodsince is not None and ifmatch is None and (st.st_mtime > unmodsince.timestamp()):
            return await self._precondition_failed(request)
        ifnonematch = request.if_none_match
        if ifnonematch is not None and self._strong_etag_match(etag_value, ifnonematch):
            return await self._not_modified(request, etag_value, last_modified)
        modsince = request.if_modified_since
        if modsince is not None and ifnonematch is None and (st.st_mtime <= modsince.timestamp()):
            return await self._not_modified(request, etag_value, last_modified)
        ct = None
        if hdrs.CONTENT_TYPE not in self.headers:
            (ct, encoding) = mimetypes.guess_type(str(filepath))
            if not ct:
                ct = 'application/octet-stream'
        else:
            encoding = 'gzip' if gzip else None
        status = self._status
        file_size = st.st_size
        count = file_size
        start = None
        ifrange = request.if_range
        if ifrange is None or st.st_mtime <= ifrange.timestamp():
            try:
                rng = request.http_range
                start = rng.start
                end = rng.stop
            except ValueError:
                self.headers[hdrs.CONTENT_RANGE] = f'bytes */{file_size}'
                self.set_status(HTTPRequestRangeNotSatisfiable.status_code)
                return await super().prepare(request)
            if start is not None or end is not None:
                if start < 0 and end is None:
                    start += file_size
                    if start < 0:
                        start = 0
                    count = file_size - start
                else:
                    count = min(end if end is not None else file_size, file_size) - start
                if start >= file_size:
                    self.headers[hdrs.CONTENT_RANGE] = f'bytes */{file_size}'
                    self.set_status(HTTPRequestRangeNotSatisfiable.status_code)
                    return await super().prepare(request)
                status = HTTPPartialContent.status_code
                self.set_status(status)
        if ct:
            self.content_type = ct
        if encoding:
            self.headers[hdrs.CONTENT_ENCODING] = encoding
        if gzip:
            self.headers[hdrs.VARY] = hdrs.ACCEPT_ENCODING
        self.etag = etag_value
        self.last_modified = st.st_mtime
        self.content_length = count
        self.headers[hdrs.ACCEPT_RANGES] = 'bytes'
        real_start = cast(int, start)
        if status == HTTPPartialContent.status_code:
            self.headers[hdrs.CONTENT_RANGE] = 'bytes {}-{}/{}'.format(real_start, real_start + count - 1, file_size)
        if count == 0 or must_be_empty_body(request.method, self.status):
            return await super().prepare(request)
        fobj = await loop.run_in_executor(None, filepath.open, 'rb')
        if start:
            offset = start
        else:
            offset = 0
        try:
            return await self._sendfile(request, fobj, offset, count)
        finally:
            await asyncio.shield(loop.run_in_executor(None, fobj.close))