"""Integration test for Python cross-language pipelines for Java KafkaIO."""
import contextlib
import logging
import os
import socket
import subprocess
import sys
import time
import typing
import unittest
import uuid
import pytest
import apache_beam as beam
from apache_beam.coders.coders import VarIntCoder
from apache_beam.io.kafka import ReadFromKafka
from apache_beam.io.kafka import WriteToKafka
from apache_beam.metrics import Metrics
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to
from apache_beam.transforms.userstate import BagStateSpec
from apache_beam.transforms.userstate import CombiningValueStateSpec
NUM_RECORDS = 1000

class CollectingFn(beam.DoFn):
    BUFFER_STATE = BagStateSpec('buffer', VarIntCoder())
    COUNT_STATE = CombiningValueStateSpec('count', sum)

    def process(self, element, buffer_state=beam.DoFn.StateParam(BUFFER_STATE), count_state=beam.DoFn.StateParam(COUNT_STATE)):
        if False:
            return 10
        value = int(element[1].decode())
        buffer_state.add(value)
        count_state.add(1)
        count = count_state.read()
        if count >= NUM_RECORDS:
            yield sum(buffer_state.read())
            count_state.clear()
            buffer_state.clear()

class CrossLanguageKafkaIO(object):

    def __init__(self, bootstrap_servers, topic, null_key, expansion_service=None):
        if False:
            while True:
                i = 10
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.null_key = null_key
        self.expansion_service = expansion_service
        self.sum_counter = Metrics.counter('source', 'elements_sum')

    def build_write_pipeline(self, pipeline):
        if False:
            while True:
                i = 10
        _ = pipeline | 'Generate' >> beam.Create(range(NUM_RECORDS)) | 'MakeKV' >> beam.Map(lambda x: (None if self.null_key else b'key', str(x).encode())).with_output_types(typing.Tuple[typing.Optional[bytes], bytes]) | 'WriteToKafka' >> WriteToKafka(producer_config={'bootstrap.servers': self.bootstrap_servers}, topic=self.topic, expansion_service=self.expansion_service)

    def build_read_pipeline(self, pipeline, max_num_records=None):
        if False:
            i = 10
            return i + 15
        kafka_records = pipeline | 'ReadFromKafka' >> ReadFromKafka(consumer_config={'bootstrap.servers': self.bootstrap_servers, 'auto.offset.reset': 'earliest'}, topics=[self.topic], max_num_records=max_num_records, expansion_service=self.expansion_service)
        if max_num_records:
            return kafka_records
        return kafka_records | 'CalculateSum' >> beam.ParDo(CollectingFn()) | 'SetSumCounter' >> beam.Map(self.sum_counter.inc)

    def run_xlang_kafkaio(self, pipeline):
        if False:
            for i in range(10):
                print('nop')
        self.build_write_pipeline(pipeline)
        self.build_read_pipeline(pipeline)
        pipeline.run(False)

class CrossLanguageKafkaIOTest(unittest.TestCase):

    @unittest.skipUnless(os.environ.get('LOCAL_KAFKA_JAR'), 'LOCAL_KAFKA_JAR environment var is not provided.')
    def test_local_kafkaio_populated_key(self):
        if False:
            i = 10
            return i + 15
        kafka_topic = 'xlang_kafkaio_test_populated_key_{}'.format(uuid.uuid4())
        local_kafka_jar = os.environ.get('LOCAL_KAFKA_JAR')
        with self.local_kafka_service(local_kafka_jar) as kafka_port:
            bootstrap_servers = '{}:{}'.format(self.get_platform_localhost(), kafka_port)
            pipeline_creator = CrossLanguageKafkaIO(bootstrap_servers, kafka_topic, False)
            self.run_kafka_write(pipeline_creator)
            self.run_kafka_read(pipeline_creator, b'key')

    @unittest.skipUnless(os.environ.get('LOCAL_KAFKA_JAR'), 'LOCAL_KAFKA_JAR environment var is not provided.')
    def test_local_kafkaio_null_key(self):
        if False:
            return 10
        kafka_topic = 'xlang_kafkaio_test_null_key_{}'.format(uuid.uuid4())
        local_kafka_jar = os.environ.get('LOCAL_KAFKA_JAR')
        with self.local_kafka_service(local_kafka_jar) as kafka_port:
            bootstrap_servers = '{}:{}'.format(self.get_platform_localhost(), kafka_port)
            pipeline_creator = CrossLanguageKafkaIO(bootstrap_servers, kafka_topic, True)
            self.run_kafka_write(pipeline_creator)
            self.run_kafka_read(pipeline_creator, None)

    @pytest.mark.uses_io_expansion_service
    @unittest.skipUnless(os.environ.get('EXPANSION_PORT'), 'EXPANSION_PORT environment var is not provided.')
    @unittest.skipUnless(os.environ.get('KAFKA_BOOTSTRAP_SERVER'), 'KAFKA_BOOTSTRAP_SERVER environment var is not provided.')
    def test_hosted_kafkaio_populated_key(self):
        if False:
            for i in range(10):
                print('nop')
        kafka_topic = 'xlang_kafkaio_test_populated_key_{}'.format(uuid.uuid4())
        bootstrap_servers = os.environ.get('KAFKA_BOOTSTRAP_SERVER')
        pipeline_creator = CrossLanguageKafkaIO(bootstrap_servers, kafka_topic, False, 'localhost:%s' % os.environ.get('EXPANSION_PORT'))
        self.run_kafka_write(pipeline_creator)
        self.run_kafka_read(pipeline_creator, b'key')

    @pytest.mark.uses_io_expansion_service
    @unittest.skipUnless(os.environ.get('EXPANSION_PORT'), 'EXPANSION_PORT environment var is not provided.')
    @unittest.skipUnless(os.environ.get('KAFKA_BOOTSTRAP_SERVER'), 'KAFKA_BOOTSTRAP_SERVER environment var is not provided.')
    def test_hosted_kafkaio_null_key(self):
        if False:
            for i in range(10):
                print('nop')
        kafka_topic = 'xlang_kafkaio_test_null_key_{}'.format(uuid.uuid4())
        bootstrap_servers = os.environ.get('KAFKA_BOOTSTRAP_SERVER')
        pipeline_creator = CrossLanguageKafkaIO(bootstrap_servers, kafka_topic, True, 'localhost:%s' % os.environ.get('EXPANSION_PORT'))
        self.run_kafka_write(pipeline_creator)
        self.run_kafka_read(pipeline_creator, None)

    def run_kafka_write(self, pipeline_creator):
        if False:
            for i in range(10):
                print('nop')
        with TestPipeline() as pipeline:
            pipeline.not_use_test_runner_api = True
            pipeline_creator.build_write_pipeline(pipeline)

    def run_kafka_read(self, pipeline_creator, expected_key):
        if False:
            return 10
        with TestPipeline() as pipeline:
            pipeline.not_use_test_runner_api = True
            result = pipeline_creator.build_read_pipeline(pipeline, NUM_RECORDS)
            assert_that(result, equal_to([(expected_key, str(i).encode()) for i in range(NUM_RECORDS)]))

    def get_platform_localhost(self):
        if False:
            for i in range(10):
                print('nop')
        if sys.platform == 'darwin':
            return 'host.docker.internal'
        else:
            return 'localhost'

    def get_open_port(self):
        if False:
            while True:
                i = 10
        s = None
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        except:
            s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        s.bind(('localhost', 0))
        s.listen(1)
        port = s.getsockname()[1]
        s.close()
        return port

    @contextlib.contextmanager
    def local_kafka_service(self, local_kafka_jar_file):
        if False:
            print('Hello World!')
        kafka_port = str(self.get_open_port())
        zookeeper_port = str(self.get_open_port())
        kafka_server = None
        try:
            kafka_server = subprocess.Popen(['java', '-jar', local_kafka_jar_file, kafka_port, zookeeper_port])
            time.sleep(3)
            yield kafka_port
        finally:
            if kafka_server:
                kafka_server.kill()
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()