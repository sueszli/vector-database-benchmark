"""
Url Length Spider Middleware

See documentation in docs/topics/spider-middleware.rst
"""
from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Any, AsyncIterable, Iterable
from scrapy import Spider
from scrapy.exceptions import NotConfigured
from scrapy.http import Request, Response
from scrapy.settings import BaseSettings
if TYPE_CHECKING:
    from typing_extensions import Self
logger = logging.getLogger(__name__)

class UrlLengthMiddleware:

    def __init__(self, maxlength: int):
        if False:
            i = 10
            return i + 15
        self.maxlength: int = maxlength

    @classmethod
    def from_settings(cls, settings: BaseSettings) -> Self:
        if False:
            i = 10
            return i + 15
        maxlength = settings.getint('URLLENGTH_LIMIT')
        if not maxlength:
            raise NotConfigured
        return cls(maxlength)

    def process_spider_output(self, response: Response, result: Iterable[Any], spider: Spider) -> Iterable[Any]:
        if False:
            while True:
                i = 10
        return (r for r in result if self._filter(r, spider))

    async def process_spider_output_async(self, response: Response, result: AsyncIterable[Any], spider: Spider) -> AsyncIterable[Any]:
        async for r in result:
            if self._filter(r, spider):
                yield r

    def _filter(self, request: Any, spider: Spider) -> bool:
        if False:
            i = 10
            return i + 15
        if isinstance(request, Request) and len(request.url) > self.maxlength:
            logger.info('Ignoring link (url length > %(maxlength)d): %(url)s ', {'maxlength': self.maxlength, 'url': request.url}, extra={'spider': spider})
            assert spider.crawler.stats
            spider.crawler.stats.inc_value('urllength/request_ignored_count', spider=spider)
            return False
        return True