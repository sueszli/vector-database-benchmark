from __future__ import absolute_import
import pytest
import io
from kafka.client_async import KafkaClient
from kafka.cluster import ClusterMetadata
from kafka.metrics import Metrics
from kafka.protocol.produce import ProduceRequest
from kafka.producer.record_accumulator import RecordAccumulator, ProducerBatch
from kafka.producer.sender import Sender
from kafka.record.memory_records import MemoryRecordsBuilder
from kafka.structs import TopicPartition

@pytest.fixture
def client(mocker):
    if False:
        for i in range(10):
            print('nop')
    _cli = mocker.Mock(spec=KafkaClient(bootstrap_servers=(), api_version=(0, 9)))
    _cli.cluster = mocker.Mock(spec=ClusterMetadata())
    return _cli

@pytest.fixture
def accumulator():
    if False:
        return 10
    return RecordAccumulator()

@pytest.fixture
def metrics():
    if False:
        while True:
            i = 10
    return Metrics()

@pytest.fixture
def sender(client, accumulator, metrics):
    if False:
        i = 10
        return i + 15
    return Sender(client, client.cluster, accumulator, metrics)

@pytest.mark.parametrize(('api_version', 'produce_version'), [((0, 10), 2), ((0, 9), 1), ((0, 8), 0)])
def test_produce_request(sender, mocker, api_version, produce_version):
    if False:
        i = 10
        return i + 15
    sender.config['api_version'] = api_version
    tp = TopicPartition('foo', 0)
    buffer = io.BytesIO()
    records = MemoryRecordsBuilder(magic=1, compression_type=0, batch_size=100000)
    batch = ProducerBatch(tp, records, buffer)
    records.close()
    produce_request = sender._produce_request(0, 0, 0, [batch])
    assert isinstance(produce_request, ProduceRequest[produce_version])