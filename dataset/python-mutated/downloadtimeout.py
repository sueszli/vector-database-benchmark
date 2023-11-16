"""
Download timeout middleware

See documentation in docs/topics/downloader-middleware.rst
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Union
from scrapy import Request, Spider, signals
from scrapy.crawler import Crawler
from scrapy.http import Response
if TYPE_CHECKING:
    from typing_extensions import Self

class DownloadTimeoutMiddleware:

    def __init__(self, timeout: float=180):
        if False:
            return 10
        self._timeout: float = timeout

    @classmethod
    def from_crawler(cls, crawler: Crawler) -> Self:
        if False:
            print('Hello World!')
        o = cls(crawler.settings.getfloat('DOWNLOAD_TIMEOUT'))
        crawler.signals.connect(o.spider_opened, signal=signals.spider_opened)
        return o

    def spider_opened(self, spider: Spider) -> None:
        if False:
            i = 10
            return i + 15
        self._timeout = getattr(spider, 'download_timeout', self._timeout)

    def process_request(self, request: Request, spider: Spider) -> Union[Request, Response, None]:
        if False:
            return 10
        if self._timeout:
            request.meta.setdefault('download_timeout', self._timeout)
        return None