import re
from typing import Optional
from dateutil.relativedelta import relativedelta
from django.utils.timezone import now
from loginas.utils import is_impersonated_session
from rest_framework import exceptions, viewsets
from rest_framework.response import Response
from posthog.client import sync_execute
from posthog.cloud_utils import is_cloud
from posthog.settings.base_variables import DEBUG
from posthog.settings.data_stores import CLICKHOUSE_CLUSTER

class DebugCHQueries(viewsets.ViewSet):
    """
    Show recent queries for this user
    """

    def _get_path(self, query: str) -> Optional[str]:
        if False:
            while True:
                i = 10
        try:
            return re.findall('request:([a-zA-Z0-9-_@]+)', query)[0].replace('_', '/')
        except:
            return None

    def list(self, request):
        if False:
            print('Hello World!')
        if not (request.user.is_staff or DEBUG or is_impersonated_session(request) or (not is_cloud())):
            raise exceptions.PermissionDenied("You're not allowed to see queries.")
        response = sync_execute('\n            select\n                query, query_start_time, exception, toInt8(type), query_duration_ms\n            from clusterAllReplicas(%(cluster)s, system, query_log)\n            where\n                query LIKE %(query)s and\n                query_start_time > %(start_time)s and\n                type != 1 and\n                query not like %(not_query)s\n            order by query_start_time desc\n            limit 100', {'query': f'/* user_id:{request.user.pk} %', 'start_time': (now() - relativedelta(minutes=10)).timestamp(), 'not_query': '%request:_api_debug_ch_queries_%', 'cluster': CLICKHOUSE_CLUSTER})
        return Response([{'query': resp[0], 'timestamp': resp[1], 'exception': resp[2], 'type': resp[3], 'execution_time': resp[4], 'path': self._get_path(resp[0])} for resp in response])