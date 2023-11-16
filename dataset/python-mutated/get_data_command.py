import logging
from typing import Any
from flask_babel import gettext as _
from superset.charts.commands.exceptions import ChartDataCacheLoadError, ChartDataQueryFailedError
from superset.commands.base import BaseCommand
from superset.common.query_context import QueryContext
from superset.exceptions import CacheLoadError
logger = logging.getLogger(__name__)

class ChartDataCommand(BaseCommand):
    _query_context: QueryContext

    def __init__(self, query_context: QueryContext):
        if False:
            while True:
                i = 10
        self._query_context = query_context

    def run(self, **kwargs: Any) -> dict[str, Any]:
        if False:
            print('Hello World!')
        cache_query_context = kwargs.get('cache', False)
        force_cached = kwargs.get('force_cached', False)
        try:
            payload = self._query_context.get_payload(cache_query_context=cache_query_context, force_cached=force_cached)
        except CacheLoadError as ex:
            raise ChartDataCacheLoadError(ex.message) from ex
        for query in payload['queries']:
            if query.get('error'):
                raise ChartDataQueryFailedError(_('Error: %(error)s', error=query['error']))
        return_value = {'query_context': self._query_context, 'queries': payload['queries']}
        if cache_query_context:
            return_value.update(cache_key=payload['cache_key'])
        return return_value

    def validate(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._query_context.raise_for_access()