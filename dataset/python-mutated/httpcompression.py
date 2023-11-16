from __future__ import annotations
import io
import zlib
from typing import TYPE_CHECKING, List, Optional, Union
from scrapy import Request, Spider
from scrapy.crawler import Crawler
from scrapy.exceptions import NotConfigured
from scrapy.http import Response, TextResponse
from scrapy.responsetypes import responsetypes
from scrapy.statscollectors import StatsCollector
from scrapy.utils.gz import gunzip
if TYPE_CHECKING:
    from typing_extensions import Self
ACCEPTED_ENCODINGS: List[bytes] = [b'gzip', b'deflate']
try:
    import brotli
    ACCEPTED_ENCODINGS.append(b'br')
except ImportError:
    pass
try:
    import zstandard
    ACCEPTED_ENCODINGS.append(b'zstd')
except ImportError:
    pass

class HttpCompressionMiddleware:
    """This middleware allows compressed (gzip, deflate) traffic to be
    sent/received from web sites"""

    def __init__(self, stats: Optional[StatsCollector]=None):
        if False:
            i = 10
            return i + 15
        self.stats = stats

    @classmethod
    def from_crawler(cls, crawler: Crawler) -> Self:
        if False:
            return 10
        if not crawler.settings.getbool('COMPRESSION_ENABLED'):
            raise NotConfigured
        return cls(stats=crawler.stats)

    def process_request(self, request: Request, spider: Spider) -> Union[Request, Response, None]:
        if False:
            i = 10
            return i + 15
        request.headers.setdefault('Accept-Encoding', b', '.join(ACCEPTED_ENCODINGS))
        return None

    def process_response(self, request: Request, response: Response, spider: Spider) -> Union[Request, Response]:
        if False:
            for i in range(10):
                print('nop')
        if request.method == 'HEAD':
            return response
        if isinstance(response, Response):
            content_encoding = response.headers.getlist('Content-Encoding')
            if content_encoding:
                encoding = content_encoding.pop()
                decoded_body = self._decode(response.body, encoding.lower())
                if self.stats:
                    self.stats.inc_value('httpcompression/response_bytes', len(decoded_body), spider=spider)
                    self.stats.inc_value('httpcompression/response_count', spider=spider)
                respcls = responsetypes.from_args(headers=response.headers, url=response.url, body=decoded_body)
                kwargs = dict(cls=respcls, body=decoded_body)
                if issubclass(respcls, TextResponse):
                    kwargs['encoding'] = None
                response = response.replace(**kwargs)
                if not content_encoding:
                    del response.headers['Content-Encoding']
        return response

    def _decode(self, body: bytes, encoding: bytes) -> bytes:
        if False:
            i = 10
            return i + 15
        if encoding == b'gzip' or encoding == b'x-gzip':
            body = gunzip(body)
        if encoding == b'deflate':
            try:
                body = zlib.decompress(body)
            except zlib.error:
                body = zlib.decompress(body, -15)
        if encoding == b'br' and b'br' in ACCEPTED_ENCODINGS:
            body = brotli.decompress(body)
        if encoding == b'zstd' and b'zstd' in ACCEPTED_ENCODINGS:
            reader = zstandard.ZstdDecompressor().stream_reader(io.BytesIO(body))
            body = reader.read()
        return body