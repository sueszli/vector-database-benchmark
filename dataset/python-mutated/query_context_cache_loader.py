from typing import Any
from superset import cache
from superset.charts.commands.exceptions import ChartDataCacheLoadError

class QueryContextCacheLoader:

    @staticmethod
    def load(cache_key: str) -> dict[str, Any]:
        if False:
            while True:
                i = 10
        cache_value = cache.get(cache_key)
        if not cache_value:
            raise ChartDataCacheLoadError('Cached data not found')
        return cache_value['data']