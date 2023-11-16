from st2common.util.monkey_patch import monkey_patch
monkey_patch()
from kombu import Exchange
from kombu.serialization import pickle
import os
import json
import pytest
import zstandard as zstd
from st2common.models.db.liveaction import LiveActionDB
from st2common.transport import publishers
from common import FIXTURES_DIR
from common import PYTEST_FIXTURE_FILE_PARAM_DECORATOR

@PYTEST_FIXTURE_FILE_PARAM_DECORATOR
@pytest.mark.parametrize('algorithm', ['none', 'zstandard'], ids=['none', 'zstandard'])
@pytest.mark.benchmark(group='no_publish')
def test_pickled_object_compression(benchmark, fixture_file: str, algorithm: str) -> None:
    if False:
        print('Hello World!')
    with open(os.path.join(FIXTURES_DIR, fixture_file), 'r') as fp:
        content = fp.read()
    data = json.loads(content)

    def run_benchmark():
        if False:
            i = 10
            return i + 15
        live_action_db = LiveActionDB()
        live_action_db.status = 'succeeded'
        live_action_db.action = 'core.local'
        live_action_db.result = data
        serialized = pickle.dumps(live_action_db)
        if algorithm == 'zstandard':
            c = zstd.ZstdCompressor()
            serialized = c.compress(serialized)
        return serialized
    result = benchmark.pedantic(run_benchmark, iterations=5, rounds=5)
    assert isinstance(result, bytes)

@PYTEST_FIXTURE_FILE_PARAM_DECORATOR
@pytest.mark.parametrize('algorithm', ['none', 'zstandard'], ids=['none', 'zstandard'])
@pytest.mark.benchmark(group='publish')
def test_pickled_object_compression_publish(benchmark, fixture_file: str, algorithm: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    with open(os.path.join(FIXTURES_DIR, fixture_file), 'r') as fp:
        content = fp.read()
    data = json.loads(content)
    publisher = publishers.PoolPublisher()
    exchange = Exchange('st2.execution.test', type='topic')
    if algorithm == 'zstandard':
        compression = 'zstd'
    else:
        compression = None

    def run_benchmark():
        if False:
            for i in range(10):
                print('nop')
        live_action_db = LiveActionDB()
        live_action_db.status = 'succeeded'
        live_action_db.action = 'core.local'
        live_action_db.result = data
        publisher.publish(payload=live_action_db, exchange=exchange, compression=compression)
    benchmark.pedantic(run_benchmark, iterations=5, rounds=5)