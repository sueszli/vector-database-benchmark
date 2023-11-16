"""
Scrapy extension for collecting scraping stats
"""
import logging
import pprint
from typing import TYPE_CHECKING, Any, Dict, Optional
from scrapy import Spider
if TYPE_CHECKING:
    from scrapy.crawler import Crawler
logger = logging.getLogger(__name__)
StatsT = Dict[str, Any]

class StatsCollector:

    def __init__(self, crawler: 'Crawler'):
        if False:
            print('Hello World!')
        self._dump: bool = crawler.settings.getbool('STATS_DUMP')
        self._stats: StatsT = {}

    def get_value(self, key: str, default: Any=None, spider: Optional[Spider]=None) -> Any:
        if False:
            i = 10
            return i + 15
        return self._stats.get(key, default)

    def get_stats(self, spider: Optional[Spider]=None) -> StatsT:
        if False:
            i = 10
            return i + 15
        return self._stats

    def set_value(self, key: str, value: Any, spider: Optional[Spider]=None) -> None:
        if False:
            i = 10
            return i + 15
        self._stats[key] = value

    def set_stats(self, stats: StatsT, spider: Optional[Spider]=None) -> None:
        if False:
            return 10
        self._stats = stats

    def inc_value(self, key: str, count: int=1, start: int=0, spider: Optional[Spider]=None) -> None:
        if False:
            while True:
                i = 10
        d = self._stats
        d[key] = d.setdefault(key, start) + count

    def max_value(self, key: str, value: Any, spider: Optional[Spider]=None) -> None:
        if False:
            print('Hello World!')
        self._stats[key] = max(self._stats.setdefault(key, value), value)

    def min_value(self, key: str, value: Any, spider: Optional[Spider]=None) -> None:
        if False:
            print('Hello World!')
        self._stats[key] = min(self._stats.setdefault(key, value), value)

    def clear_stats(self, spider: Optional[Spider]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._stats.clear()

    def open_spider(self, spider: Spider) -> None:
        if False:
            while True:
                i = 10
        pass

    def close_spider(self, spider: Spider, reason: str) -> None:
        if False:
            print('Hello World!')
        if self._dump:
            logger.info('Dumping Scrapy stats:\n' + pprint.pformat(self._stats), extra={'spider': spider})
        self._persist_stats(self._stats, spider)

    def _persist_stats(self, stats: StatsT, spider: Spider) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

class MemoryStatsCollector(StatsCollector):

    def __init__(self, crawler: 'Crawler'):
        if False:
            while True:
                i = 10
        super().__init__(crawler)
        self.spider_stats: Dict[str, StatsT] = {}

    def _persist_stats(self, stats: StatsT, spider: Spider) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.spider_stats[spider.name] = stats

class DummyStatsCollector(StatsCollector):

    def get_value(self, key: str, default: Any=None, spider: Optional[Spider]=None) -> Any:
        if False:
            while True:
                i = 10
        return default

    def set_value(self, key: str, value: Any, spider: Optional[Spider]=None) -> None:
        if False:
            print('Hello World!')
        pass

    def set_stats(self, stats: StatsT, spider: Optional[Spider]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    def inc_value(self, key: str, count: int=1, start: int=0, spider: Optional[Spider]=None) -> None:
        if False:
            return 10
        pass

    def max_value(self, key: str, value: Any, spider: Optional[Spider]=None) -> None:
        if False:
            i = 10
            return i + 15
        pass

    def min_value(self, key: str, value: Any, spider: Optional[Spider]=None) -> None:
        if False:
            while True:
                i = 10
        pass