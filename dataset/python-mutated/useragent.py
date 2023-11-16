"""Set User-Agent header per spider or use a default value from settings"""
from __future__ import annotations
from typing import TYPE_CHECKING, Union
from scrapy import Request, Spider, signals
from scrapy.crawler import Crawler
from scrapy.http import Response
if TYPE_CHECKING:
    from typing_extensions import Self

class UserAgentMiddleware:
    """This middleware allows spiders to override the user_agent"""

    def __init__(self, user_agent: str='Scrapy'):
        if False:
            for i in range(10):
                print('nop')
        self.user_agent = user_agent

    @classmethod
    def from_crawler(cls, crawler: Crawler) -> Self:
        if False:
            i = 10
            return i + 15
        o = cls(crawler.settings['USER_AGENT'])
        crawler.signals.connect(o.spider_opened, signal=signals.spider_opened)
        return o

    def spider_opened(self, spider: Spider) -> None:
        if False:
            while True:
                i = 10
        self.user_agent = getattr(spider, 'user_agent', self.user_agent)

    def process_request(self, request: Request, spider: Spider) -> Union[Request, Response, None]:
        if False:
            return 10
        if self.user_agent:
            request.headers.setdefault(b'User-Agent', self.user_agent)
        return None