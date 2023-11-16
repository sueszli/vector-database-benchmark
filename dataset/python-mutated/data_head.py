import json
import os
from enum import Enum
import aiohttp
from aiohttp.web import Request, Response
import ray.dashboard.optional_utils as optional_utils
import ray.dashboard.utils as dashboard_utils
from ray.dashboard.modules.metrics.metrics_head import PROMETHEUS_HOST_ENV_VAR, DEFAULT_PROMETHEUS_HOST, PrometheusQueryError
from urllib.parse import quote
import ray
MAX_TIME_WINDOW = '1h'
SAMPLE_RATE = '1s'

class PrometheusQuery(Enum):
    """Enum to store types of Prometheus queries for a given metric and grouping."""
    VALUE = ('value', 'sum({}) by ({})')
    MAX = ('max', 'max_over_time(sum({}) by ({})[' + f'{MAX_TIME_WINDOW}:{SAMPLE_RATE}])')
DATASET_METRICS = {'ray_data_output_bytes': (PrometheusQuery.MAX,), 'ray_data_spilled_bytes': (PrometheusQuery.MAX,), 'ray_data_current_bytes': (PrometheusQuery.VALUE, PrometheusQuery.MAX)}

class DataHead(dashboard_utils.DashboardHeadModule):

    def __init__(self, dashboard_head):
        if False:
            print('Hello World!')
        super().__init__(dashboard_head)
        self.http_session = aiohttp.ClientSession()
        self.prometheus_host = os.environ.get(PROMETHEUS_HOST_ENV_VAR, DEFAULT_PROMETHEUS_HOST)

    @optional_utils.DashboardHeadRouteTable.get('/api/data/datasets')
    @optional_utils.init_ray_and_catch_exceptions()
    async def get_datasets(self, req: Request) -> Response:
        try:
            from ray.data._internal.stats import _get_or_create_stats_actor
            _stats_actor = _get_or_create_stats_actor()
            datasets = ray.get(_stats_actor.get_datasets.remote())
            for dataset in datasets:
                for (metric, queries) in DATASET_METRICS.items():
                    datasets[dataset][metric] = {query.value[0]: 0 for query in queries}
                    for operator in datasets[dataset]['operators']:
                        datasets[dataset]['operators'][operator][metric] = {query.value[0]: 0 for query in queries}
            try:
                for (metric, queries) in DATASET_METRICS.items():
                    for query in queries:
                        (query_name, prom_query) = query.value
                        dataset_result = await self._query_prometheus(prom_query.format(metric, 'dataset'))
                        for res in dataset_result['data']['result']:
                            (dataset, value) = (res['metric']['dataset'], res['value'][1])
                            if dataset in datasets:
                                datasets[dataset][metric][query_name] = value
                        operator_result = await self._query_prometheus(prom_query.format(metric, 'dataset, operator'))
                        for res in operator_result['data']['result']:
                            (dataset, operator, value) = (res['metric']['dataset'], res['metric']['operator'], res['value'][1])
                            if dataset in datasets and operator in datasets[dataset]['operators']:
                                datasets[dataset]['operators'][operator][metric][query_name] = value
            except aiohttp.client_exceptions.ClientConnectorError:
                pass
            for dataset in datasets:
                datasets[dataset]['operators'] = list(map(lambda item: {'operator': item[0], **item[1]}, datasets[dataset]['operators'].items()))
            datasets = list(map(lambda item: {'dataset': item[0], **item[1]}, datasets.items()))
            datasets = sorted(datasets, key=lambda x: x['start_time'], reverse=True)
            return Response(text=json.dumps({'datasets': datasets}), content_type='application/json')
        except Exception as e:
            return Response(status=503, text=str(e))

    async def run(self, server):
        pass

    @staticmethod
    def is_minimal_module():
        if False:
            for i in range(10):
                print('nop')
        return False

    async def _query_prometheus(self, query):
        async with self.http_session.get(f'{self.prometheus_host}/api/v1/query?query={quote(query)}') as resp:
            if resp.status == 200:
                prom_data = await resp.json()
                return prom_data
            message = await resp.text()
            raise PrometheusQueryError(resp.status, message)