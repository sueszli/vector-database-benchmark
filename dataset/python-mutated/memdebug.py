"""
MemoryDebugger extension

See documentation in docs/topics/extensions.rst
"""
import gc
from scrapy import signals
from scrapy.exceptions import NotConfigured
from scrapy.utils.trackref import live_refs

class MemoryDebugger:

    def __init__(self, stats):
        if False:
            print('Hello World!')
        self.stats = stats

    @classmethod
    def from_crawler(cls, crawler):
        if False:
            return 10
        if not crawler.settings.getbool('MEMDEBUG_ENABLED'):
            raise NotConfigured
        o = cls(crawler.stats)
        crawler.signals.connect(o.spider_closed, signal=signals.spider_closed)
        return o

    def spider_closed(self, spider, reason):
        if False:
            i = 10
            return i + 15
        gc.collect()
        self.stats.set_value('memdebug/gc_garbage_count', len(gc.garbage), spider=spider)
        for (cls, wdict) in live_refs.items():
            if not wdict:
                continue
            self.stats.set_value(f'memdebug/live_refs/{cls.__name__}', len(wdict), spider=spider)