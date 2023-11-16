"""
Offsite Spider Middleware

See documentation in docs/topics/spider-middleware.rst
"""
from __future__ import annotations
import logging
import re
import warnings
from typing import TYPE_CHECKING, Any, AsyncIterable, Iterable, Set
from scrapy import Spider, signals
from scrapy.crawler import Crawler
from scrapy.http import Request, Response
from scrapy.statscollectors import StatsCollector
from scrapy.utils.httpobj import urlparse_cached
if TYPE_CHECKING:
    from typing_extensions import Self
logger = logging.getLogger(__name__)

class OffsiteMiddleware:

    def __init__(self, stats: StatsCollector):
        if False:
            return 10
        self.stats: StatsCollector = stats

    @classmethod
    def from_crawler(cls, crawler: Crawler) -> Self:
        if False:
            for i in range(10):
                print('nop')
        assert crawler.stats
        o = cls(crawler.stats)
        crawler.signals.connect(o.spider_opened, signal=signals.spider_opened)
        return o

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
            for i in range(10):
                print('nop')
        if not isinstance(request, Request):
            return True
        if request.dont_filter or self.should_follow(request, spider):
            return True
        domain = urlparse_cached(request).hostname
        if domain and domain not in self.domains_seen:
            self.domains_seen.add(domain)
            logger.debug('Filtered offsite request to %(domain)r: %(request)s', {'domain': domain, 'request': request}, extra={'spider': spider})
            self.stats.inc_value('offsite/domains', spider=spider)
        self.stats.inc_value('offsite/filtered', spider=spider)
        return False

    def should_follow(self, request: Request, spider: Spider) -> bool:
        if False:
            return 10
        regex = self.host_regex
        host = urlparse_cached(request).hostname or ''
        return bool(regex.search(host))

    def get_host_regex(self, spider: Spider) -> re.Pattern[str]:
        if False:
            i = 10
            return i + 15
        'Override this method to implement a different offsite policy'
        allowed_domains = getattr(spider, 'allowed_domains', None)
        if not allowed_domains:
            return re.compile('')
        url_pattern = re.compile('^https?://.*$')
        port_pattern = re.compile(':\\d+$')
        domains = []
        for domain in allowed_domains:
            if domain is None:
                continue
            if url_pattern.match(domain):
                message = f'allowed_domains accepts only domains, not URLs. Ignoring URL entry {domain} in allowed_domains.'
                warnings.warn(message, URLWarning)
            elif port_pattern.search(domain):
                message = f'allowed_domains accepts only domains without ports. Ignoring entry {domain} in allowed_domains.'
                warnings.warn(message, PortWarning)
            else:
                domains.append(re.escape(domain))
        regex = f"^(.*\\.)?({'|'.join(domains)})$"
        return re.compile(regex)

    def spider_opened(self, spider: Spider) -> None:
        if False:
            while True:
                i = 10
        self.host_regex: re.Pattern[str] = self.get_host_regex(spider)
        self.domains_seen: Set[str] = set()

class URLWarning(Warning):
    pass

class PortWarning(Warning):
    pass