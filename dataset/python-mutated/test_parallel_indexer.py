from datetime import datetime, timezone
import pytest
from arroyo.backends.kafka import KafkaPayload
from arroyo.types import BrokerValue, Message, Partition, Topic
from sentry.metrics.middleware import global_tags
from sentry.sentry_metrics.consumers.indexer.parallel import MetricsConsumerStrategyFactory
from sentry.snuba.metrics.naming_layer.mri import SessionMRI
from sentry.utils import json
ts = int(datetime.now(tz=timezone.utc).timestamp())
counter_payload = {'name': SessionMRI.RAW_SESSION.value, 'tags': {'environment': 'production', 'session.status': 'init'}, 'timestamp': ts, 'type': 'c', 'value': 1.0, 'org_id': 1, 'project_id': 3}

@pytest.fixture(autouse=True)
def reset_global_metrics_state():
    if False:
        while True:
            i = 10
    with global_tags(_all_threads=True):
        yield

@pytest.mark.django_db
@pytest.mark.parametrize('force_disable_multiprocessing', [True, False])
def test_basic(request, settings, force_disable_multiprocessing):
    if False:
        print('Hello World!')
    '\n    Integration test to verify that the parallel indexer can spawn subprocesses\n    properly. The main purpose is to verify that there are no\n    pickling/unpickling errors when passing the strategy into the\n    ParallelTransformStep, as that is easy to break.\n    '
    settings.KAFKA_CONSUMER_FORCE_DISABLE_MULTIPROCESSING = force_disable_multiprocessing
    processing_factory = MetricsConsumerStrategyFactory(max_msg_batch_size=1, max_msg_batch_time=1, max_parallel_batch_size=1, max_parallel_batch_time=1, processes=1, input_block_size=1024, output_block_size=1024, ingest_profile='release-health', indexer_db='postgres')
    strategy = processing_factory.create_with_partitions(lambda _, force=False: None, {Partition(topic=Topic(name='ingest-bogus-metrics'), index=1): 1})
    message = Message(BrokerValue(KafkaPayload(None, json.dumps(counter_payload).encode('utf-8'), []), Partition(Topic('topic'), 0), 0, datetime.now()))
    strategy.submit(message=message)
    strategy.close()
    strategy.join()