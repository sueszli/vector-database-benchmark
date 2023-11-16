import os
import time
import sys
import pytest
import ray._private.prometheus_exporter as prometheus_exporter
from typing import List
from opencensus.stats.view_manager import ViewManager
from opencensus.stats.stats_recorder import StatsRecorder
from opencensus.stats import execution_context
from prometheus_client.core import REGISTRY
from ray._private.metrics_agent import Gauge, MetricsAgent, Record, RAY_WORKER_TIMEOUT_S
from ray._private.services import new_port
from ray.core.generated.metrics_pb2 import Metric, MetricDescriptor, Point, LabelKey, TimeSeries, LabelValue
from ray._raylet import WorkerID
from ray._private.test_utils import fetch_prometheus_metrics, fetch_raw_prometheus, wait_for_condition

def raw_metrics(export_port):
    if False:
        for i in range(10):
            print('nop')
    metrics_page = 'localhost:{}'.format(export_port)
    res = fetch_prometheus_metrics([metrics_page])
    return res

def get_metric(metric_name, export_port):
    if False:
        print('Hello World!')
    res = raw_metrics(export_port)
    for (name, samples) in res.items():
        if name == metric_name:
            return (name, samples)
    return None

def get_prom_metric_name(namespace, metric_name):
    if False:
        i = 10
        return i + 15
    return f'{namespace}_{metric_name}'

def generate_timeseries(label_values: List[str], points: List[float]):
    if False:
        i = 10
        return i + 15
    return TimeSeries(label_values=[LabelValue(value=val) for val in label_values], points=[Point(double_value=val) for val in points])

def generate_protobuf_metric(name: str, desc: str, unit: str, label_keys: List[str]=None, timeseries: List[TimeSeries]=None):
    if False:
        return 10
    if not label_keys:
        label_keys = []
    if not timeseries:
        timeseries = []
    return Metric(metric_descriptor=MetricDescriptor(name=name, description=desc, unit=unit, label_keys=[LabelKey(key='a'), LabelKey(key='b')]), timeseries=timeseries)

@pytest.fixture
def get_agent(request, monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    with monkeypatch.context() as m:
        if hasattr(request, 'param'):
            delay = request.param
        else:
            delay = 0
        m.setenv(RAY_WORKER_TIMEOUT_S, delay)
        agent_port = new_port()
        stats_recorder = StatsRecorder()
        view_manager = ViewManager()
        stats_exporter = prometheus_exporter.new_stats_exporter(prometheus_exporter.Options(namespace='test', port=agent_port, address='127.0.0.1'))
        agent = MetricsAgent(view_manager, stats_recorder, stats_exporter)
        REGISTRY.register(agent.proxy_exporter_collector)
        yield (agent, agent_port)
        REGISTRY.unregister(agent.stats_exporter.collector)
        REGISTRY.unregister(agent.proxy_exporter_collector)
        execution_context.set_measure_to_view_map({})

@pytest.mark.skipif(sys.platform == 'win32', reason='Flaky on Windows.')
def test_metrics_agent_record_and_export(get_agent):
    if False:
        print('Hello World!')
    namespace = 'test'
    (agent, agent_port) = get_agent
    metric_name = 'test'
    test_gauge = Gauge(metric_name, 'desc', 'unit', ['tag'])
    record_a = Record(gauge=test_gauge, value=3, tags={'tag': 'a'})
    agent.record_and_export([record_a])
    (name, samples) = get_metric(get_prom_metric_name(namespace, metric_name), agent_port)
    assert name == get_prom_metric_name(namespace, metric_name)
    assert len(samples) == 1
    assert samples[0].value == 3
    assert samples[0].labels == {'tag': 'a'}
    record_b = Record(gauge=test_gauge, value=4, tags={'tag': 'a'})
    record_c = Record(gauge=test_gauge, value=4, tags={'tag': 'a'})
    agent.record_and_export([record_b, record_c])
    (name, samples) = get_metric(get_prom_metric_name(namespace, metric_name), agent_port)
    assert name == get_prom_metric_name(namespace, metric_name)
    assert len(samples) == 1
    assert samples[0].value == 4
    assert samples[0].labels == {'tag': 'a'}
    record_d = Record(gauge=test_gauge, value=6, tags={'tag': 'aa'})
    agent.record_and_export([record_d])
    (name, samples) = get_metric(get_prom_metric_name(namespace, metric_name), agent_port)
    assert name == get_prom_metric_name(namespace, metric_name)
    assert len(samples) == 2
    assert samples[0].value == 4
    assert samples[0].labels == {'tag': 'a'}
    assert samples[1].value == 6
    assert samples[1].labels == {'tag': 'aa'}
    metric_name_2 = 'test2'
    test_gauge_2 = Gauge(metric_name_2, 'desc', 'unit', ['tag'])
    record_e = Record(gauge=test_gauge_2, value=1, tags={'tag': 'b'})
    agent.record_and_export([record_e])
    (name, samples) = get_metric(get_prom_metric_name(namespace, metric_name_2), agent_port)
    assert name == get_prom_metric_name(namespace, metric_name_2)
    assert samples[0].value == 1
    assert samples[0].labels == {'tag': 'b'}
    (name, samples) = get_metric(get_prom_metric_name(namespace, metric_name), agent_port)
    assert name == get_prom_metric_name(namespace, metric_name)
    assert len(samples) == 2
    assert samples[0].value == 4
    assert samples[0].labels == {'tag': 'a'}
    assert samples[1].value == 6
    assert samples[1].labels == {'tag': 'aa'}

@pytest.mark.skipif(sys.platform == 'win32', reason='Flaky on Windows.')
def test_metrics_agent_proxy_record_and_export_basic(get_agent):
    if False:
        print('Hello World!')
    'Test the case the metrics are exported without worker_id.'
    namespace = 'test'
    (agent, agent_port) = get_agent
    m = generate_protobuf_metric('test', 'desc', '', label_keys=['a', 'b'], timeseries=[])
    m.timeseries.append(generate_timeseries(['a', 'b'], [1, 2, 3]))
    agent.proxy_export_metrics([m])
    (name, samples) = get_metric(f'{namespace}_test', agent_port)
    assert name == f'{namespace}_test'
    assert len(samples) == 1
    assert samples[0].labels == {'a': 'a', 'b': 'b'}
    assert samples[0].value == 3
    m = generate_protobuf_metric('test', 'desc', '', label_keys=['a', 'b'], timeseries=[])
    m.timeseries.append(generate_timeseries(['a', 'b'], [4]))
    agent.proxy_export_metrics([m])
    (name, samples) = get_metric(f'{namespace}_test', agent_port)
    assert name == f'{namespace}_test'
    assert len(samples) == 1
    assert samples[0].labels == {'a': 'a', 'b': 'b'}
    assert samples[0].value == 4
    m = generate_protobuf_metric('test', 'desc', '', label_keys=['a', 'b'], timeseries=[])
    m.timeseries.append(generate_timeseries(['a', 'c'], [5]))
    agent.proxy_export_metrics([m])
    (name, samples) = get_metric(f'{namespace}_test', agent_port)
    assert name == f'{namespace}_test'
    assert len(samples) == 2
    assert samples[0].labels == {'a': 'a', 'b': 'b'}
    assert samples[0].value == 4
    assert samples[1].labels == {'a': 'a', 'b': 'c'}
    assert samples[1].value == 5

@pytest.mark.skipif(sys.platform == 'win32', reason='Flaky on Windows.')
def test_metrics_agent_proxy_record_and_export_from_workers(get_agent):
    if False:
        print('Hello World!')
    '\n    Test the basic worker death case.\n    '
    namespace = 'test'
    (agent, agent_port) = get_agent
    worker_id = WorkerID.from_random()
    m = generate_protobuf_metric('test', 'desc', '', label_keys=['a', 'b'], timeseries=[])
    m.timeseries.append(generate_timeseries(['a', 'b'], [1, 2, 3]))
    agent.proxy_export_metrics([m], worker_id_hex=worker_id.hex())
    assert get_metric(f'{namespace}_test', agent_port) is not None
    agent.clean_all_dead_worker_metrics()
    assert get_metric(f'{namespace}_test', agent_port) is None
    agent.proxy_export_metrics([m], worker_id_hex=worker_id.hex())
    assert get_metric(f'{namespace}_test', agent_port) is not None
    agent.clean_all_dead_worker_metrics()
    assert get_metric(f'{namespace}_test', agent_port) is None

@pytest.mark.skipif(sys.platform == 'win32', reason='Flaky on Windows.')
def test_metrics_agent_proxy_record_and_export_from_workers_complicated(get_agent):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test the complicated worker death case.\n    '
    namespace = 'test'
    (agent, agent_port) = get_agent
    worker_ids = [WorkerID.from_random() for _ in range(4)]
    metrics = []
    for i in range(8):
        m = generate_protobuf_metric(f'test_{i}', 'desc', '', label_keys=['a', 'b'], timeseries=[])
        m.timeseries.append(generate_timeseries(['a', str(i)], [3]))
        metrics.append(m)
    i = 0
    for worker_id in worker_ids:
        agent.proxy_export_metrics([metrics[i], metrics[i + 1]], worker_id_hex=worker_id.hex())
        i += 2
    for i in range(len(metrics)):
        assert get_metric(f'{namespace}_test_{i}', agent_port) is not None
    i = 0
    while len(worker_ids):
        for worker_id in worker_ids:
            agent.clean_all_dead_worker_metrics()
            assert get_metric(f'{namespace}_test_{i}', agent_port) is None
            assert get_metric(f'{namespace}_test_{i + 1}', agent_port) is None
        worker_ids.pop(0)
        metrics.pop(0)
        metrics.pop(0)
        i = 0
        for worker_id in worker_ids:
            agent.proxy_export_metrics([metrics[i], metrics[i + 1]], worker_id_hex=worker_id.hex())
            i += 2
        for i in range(i + 2, len(metrics)):
            assert get_metric(f'{namespace}_test_{i}', agent_port) is not None, i
        i += 2
DELAY = 3

@pytest.mark.skipif(sys.platform == 'win32', reason='Flaky on Windows.')
@pytest.mark.parametrize('get_agent', [DELAY], indirect=True)
def test_metrics_agent_proxy_record_and_export_from_workers_delay(get_agent):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test the worker metrics are deleted after the delay.\n    '
    namespace = 'test'
    (agent, agent_port) = get_agent
    worker_id = WorkerID.from_random()
    m = generate_protobuf_metric('test', 'desc', '', label_keys=['a', 'b'], timeseries=[])
    m.timeseries.append(generate_timeseries(['a', 'b'], [1, 2, 3]))
    agent.proxy_export_metrics([m], worker_id_hex=worker_id.hex())
    agent.clean_all_dead_worker_metrics()
    start = time.time()

    def verify():
        if False:
            for i in range(10):
                print('nop')
        agent.clean_all_dead_worker_metrics()
        return get_metric(f'{namespace}_test', agent_port) is None
    wait_for_condition(verify)
    assert time.time() - start > DELAY

@pytest.mark.skipif(sys.platform == 'win32', reason='Flaky on Windows.')
def test_metrics_agent_export_format_correct(get_agent):
    if False:
        while True:
            i = 10
    '\n    Verifies that there is one metric per metric name and not one\n    per metric name + tag combination.\n    Also verifies that the prometheus output is in the right format.\n    '
    namespace = 'test'
    (agent, agent_port) = get_agent
    metric_name = 'test'
    test_gauge = Gauge(metric_name, 'desc', 'unit', ['tag'])
    record_a = Record(gauge=test_gauge, value=3, tags={'tag': 'a'})
    agent.record_and_export([record_a])
    record_b = Record(gauge=test_gauge, value=4, tags={'tag': 'b'})
    agent.record_and_export([record_b])
    metric_name_2 = 'test2'
    test_gauge_2 = Gauge(metric_name_2, 'desc', 'unit', ['tag'])
    record_c = Record(gauge=test_gauge_2, value=1, tags={'tag': 'c'})
    agent.record_and_export([record_c])
    (name, samples) = get_metric(get_prom_metric_name(namespace, metric_name_2), agent_port)
    assert name == get_prom_metric_name(namespace, metric_name_2)
    assert len(samples) == 1
    assert samples[0].value == 1
    assert samples[0].labels == {'tag': 'c'}
    (name, samples) = get_metric(get_prom_metric_name(namespace, metric_name), agent_port)
    assert name == get_prom_metric_name(namespace, metric_name)
    assert len(samples) == 2
    assert samples[0].value == 3
    assert samples[0].labels == {'tag': 'a'}
    assert samples[1].value == 4
    assert samples[1].labels == {'tag': 'b'}
    metrics_page = 'localhost:{}'.format(agent_port)
    (_, response) = list(fetch_raw_prometheus([metrics_page]))[0]
    assert response.count('# HELP test_test desc') == 1
    assert response.count('# TYPE test_test gauge') == 1
    assert response.count('# HELP test_test2 desc') == 1
    assert response.count('# TYPE test_test2 gauge') == 1
if __name__ == '__main__':
    import sys
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))