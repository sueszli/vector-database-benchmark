from typing import Any, Dict
from litestar import Litestar, Request
from litestar.contrib.prometheus import PrometheusConfig, PrometheusController

class CustomPrometheusController(PrometheusController):
    path = '/custom-path'
    openmetrics_format = True

def custom_label_callable(request: Request[Any, Any, Any]) -> str:
    if False:
        print('Hello World!')
    return 'v2.0'
extra_labels = {'version_no': custom_label_callable, 'location': 'earth'}
buckets = [0.1, 0.2, 0.3, 0.4, 0.5]

def custom_exemplar(request: Request[Any, Any, Any]) -> Dict[str, str]:
    if False:
        print('Hello World!')
    return {'trace_id': '1234'}
prometheus_config = PrometheusConfig(app_name='litestar-example', prefix='litestar', labels=extra_labels, buckets=buckets, exemplars=custom_exemplar, excluded_http_methods=['POST'])
app = Litestar(route_handlers=[CustomPrometheusController], middleware=[prometheus_config.middleware])