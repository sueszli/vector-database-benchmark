import argparse
import logging
import shlex
import typing
import unittest
from os import linesep
from os import path
from os.path import exists
from shutil import rmtree
from tempfile import mkdtemp
import pytest
import apache_beam as beam
from apache_beam import Impulse
from apache_beam import Map
from apache_beam.io.external.generate_sequence import GenerateSequence
from apache_beam.io.kafka import ReadFromKafka
from apache_beam.io.kafka import WriteToKafka
from apache_beam.options.pipeline_options import DebugOptions
from apache_beam.options.pipeline_options import FlinkRunnerOptions
from apache_beam.options.pipeline_options import PortableOptions
from apache_beam.options.pipeline_options import StandardOptions
from apache_beam.runners.portability import job_server
from apache_beam.runners.portability import portable_runner
from apache_beam.runners.portability import portable_runner_test
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to
from apache_beam.transforms.sql import SqlTransform
_LOGGER = logging.getLogger(__name__)
Row = typing.NamedTuple('Row', [('col1', int), ('col2', str)])
beam.coders.registry.register_coder(Row, beam.coders.RowCoder)

class FlinkRunnerTest(portable_runner_test.PortableRunnerTest):
    _use_grpc = True
    _use_subprocesses = True
    conf_dir = None
    expansion_port = None
    flink_job_server_jar = None

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
        self.environment_type = None
        self.environment_config = None
        self.enable_commit = False

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.enable_commit = False

    @pytest.fixture(autouse=True)
    def parse_options(self, request):
        if False:
            while True:
                i = 10
        if not request.config.option.test_pipeline_options:
            raise unittest.SkipTest('Skipping because --test-pipeline-options is not specified.')
        test_pipeline_options = request.config.option.test_pipeline_options
        parser = argparse.ArgumentParser(add_help=True)
        parser.add_argument('--flink_job_server_jar', help='Job server jar to submit jobs.', action='store')
        parser.add_argument('--environment_type', default='LOOPBACK', choices=['DOCKER', 'PROCESS', 'LOOPBACK'], help='Set the environment type for running user code. DOCKER runs user code in a container. PROCESS runs user code in automatically started processes. LOOPBACK runs user code on the same process that originally submitted the job.')
        parser.add_argument('--environment_option', '--environment_options', dest='environment_options', action='append', default=None, help='Environment configuration for running the user code. Recognized options depend on --environment_type.\n For DOCKER: docker_container_image (optional)\n For PROCESS: process_command (required), process_variables (optional, comma-separated)\n For EXTERNAL: external_service_address (required)')
        (known_args, unknown_args) = parser.parse_known_args(shlex.split(test_pipeline_options))
        if unknown_args:
            _LOGGER.warning('Discarding unrecognized arguments %s' % unknown_args)
        self.set_flink_job_server_jar(known_args.flink_job_server_jar or job_server.JavaJarJobServer.path_to_beam_jar(':runners:flink:%s:job-server:shadowJar' % FlinkRunnerOptions.PUBLISHED_FLINK_VERSIONS[-1]))
        self.environment_type = known_args.environment_type
        self.environment_options = known_args.environment_options

    @classmethod
    def tearDownClass(cls):
        if False:
            i = 10
            return i + 15
        if cls.conf_dir and exists(cls.conf_dir):
            _LOGGER.info('removing conf dir: %s' % cls.conf_dir)
            rmtree(cls.conf_dir)
        super().tearDownClass()

    @classmethod
    def _create_conf_dir(cls):
        if False:
            while True:
                i = 10
        'Create (and save a static reference to) a "conf dir", used to provide\n     metrics configs and verify metrics output\n\n     It gets cleaned up when the suite is done executing'
        if hasattr(cls, 'conf_dir'):
            cls.conf_dir = mkdtemp(prefix='flinktest-conf')
            cls.test_metrics_path = path.join(cls.conf_dir, 'test-metrics.txt')
            conf_path = path.join(cls.conf_dir, 'flink-conf.yaml')
            file_reporter = 'org.apache.beam.runners.flink.metrics.FileReporter'
            with open(conf_path, 'w') as f:
                f.write(linesep.join(['metrics.reporters: file', 'metrics.reporter.file.class: %s' % file_reporter, 'metrics.reporter.file.path: %s' % cls.test_metrics_path, 'metrics.scope.operator: <operator_name>']))

    @classmethod
    def _subprocess_command(cls, job_port, expansion_port):
        if False:
            return 10
        tmp_dir = mkdtemp(prefix='flinktest')
        cls._create_conf_dir()
        cls.expansion_port = expansion_port
        try:
            return ['java', '-Dorg.slf4j.simpleLogger.defaultLogLevel=warn', '-jar', cls.flink_job_server_jar, '--flink-master', '[local]', '--flink-conf-dir', cls.conf_dir, '--artifacts-dir', tmp_dir, '--job-port', str(job_port), '--artifact-port', '0', '--expansion-port', str(expansion_port)]
        finally:
            rmtree(tmp_dir)

    @classmethod
    def get_runner(cls):
        if False:
            i = 10
            return i + 15
        return portable_runner.PortableRunner()

    @classmethod
    def get_expansion_service(cls):
        if False:
            for i in range(10):
                print('nop')
        return 'localhost:%s' % cls.expansion_port

    @classmethod
    def set_flink_job_server_jar(cls, flink_job_server_jar):
        if False:
            print('Hello World!')
        cls.flink_job_server_jar = flink_job_server_jar

    def create_options(self):
        if False:
            i = 10
            return i + 15
        options = super().create_options()
        options.view_as(DebugOptions).experiments = ['beam_fn_api']
        options._all_options['parallelism'] = 2
        options.view_as(PortableOptions).environment_type = self.environment_type
        options.view_as(PortableOptions).environment_options = self.environment_options
        if self.enable_commit:
            options.view_as(StandardOptions).streaming = True
            options._all_options['checkpointing_interval'] = 3000
            options._all_options['shutdown_sources_after_idle_ms'] = 60000
            options._all_options['number_of_execution_retries'] = 1
        return options

    def test_read(self):
        if False:
            for i in range(10):
                print('nop')
        print('name:', __name__)
        with self.create_pipeline() as p:
            lines = p | beam.io.ReadFromText('/etc/profile')
            assert_that(lines, lambda lines: len(lines) > 0)

    def test_no_subtransform_composite(self):
        if False:
            while True:
                i = 10
        raise unittest.SkipTest('BEAM-4781')

    def test_external_transform(self):
        if False:
            while True:
                i = 10
        with self.create_pipeline() as p:
            res = p | GenerateSequence(start=1, stop=10, expansion_service=self.get_expansion_service())
            assert_that(res, equal_to([i for i in range(1, 10)]))

    def test_expand_kafka_read(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(Exception) as ctx:
            self.enable_commit = True
            with self.create_pipeline() as p:
                p | ReadFromKafka(consumer_config={'bootstrap.servers': 'notvalid1:7777, notvalid2:3531', 'group.id': 'any_group'}, topics=['topic1', 'topic2'], key_deserializer='org.apache.kafka.common.serialization.ByteArrayDeserializer', value_deserializer='org.apache.kafka.common.serialization.LongDeserializer', commit_offset_in_finalize=True, timestamp_policy=ReadFromKafka.create_time_policy, expansion_service=self.get_expansion_service())
        self.assertTrue('No resolvable bootstrap urls given in bootstrap.servers' in str(ctx.exception), 'Expected to fail due to invalid bootstrap.servers, but failed due to:\n%s' % str(ctx.exception))

    def test_expand_kafka_write(self):
        if False:
            while True:
                i = 10
        self.create_pipeline() | Impulse() | Map(lambda input: (1, input)) | WriteToKafka(producer_config={'bootstrap.servers': 'localhost:9092, notvalid2:3531'}, topic='topic1', key_serializer='org.apache.kafka.common.serialization.LongSerializer', value_serializer='org.apache.kafka.common.serialization.ByteArraySerializer', expansion_service=self.get_expansion_service())

    def test_sql(self):
        if False:
            while True:
                i = 10
        with self.create_pipeline() as p:
            output = p | 'Create' >> beam.Create([Row(x, str(x)) for x in range(5)]) | 'Sql' >> SqlTransform("SELECT col1, col2 || '*' || col2 as col2,\n                    power(col1, 2) as col3\n             FROM PCOLLECTION\n          ", expansion_service=self.get_expansion_service())
            assert_that(output, equal_to([(x, '{x}*{x}'.format(x=x), x * x) for x in range(5)]))

    def test_flattened_side_input(self):
        if False:
            print('Hello World!')
        super().test_flattened_side_input(with_transcoding=False)

    def test_metrics(self):
        if False:
            while True:
                i = 10
        super().test_metrics(check_gauge=False)

    def test_sdf_with_watermark_tracking(self):
        if False:
            for i in range(10):
                print('nop')
        raise unittest.SkipTest('BEAM-2939')

    def test_callbacks_with_exception(self):
        if False:
            for i in range(10):
                print('nop')
        raise unittest.SkipTest('https://github.com/apache/beam/issues/19526')

    def test_register_finalizations(self):
        if False:
            while True:
                i = 10
        raise unittest.SkipTest('https://github.com/apache/beam/issues/19526')

    def test_custom_merging_window(self):
        if False:
            i = 10
            return i + 15
        raise unittest.SkipTest('https://github.com/apache/beam/issues/20641')

class FlinkRunnerTestOptimized(FlinkRunnerTest):

    def create_options(self):
        if False:
            for i in range(10):
                print('nop')
        options = super().create_options()
        options.view_as(DebugOptions).experiments = ['pre_optimize=all'] + options.view_as(DebugOptions).experiments
        return options

    def test_external_transform(self):
        if False:
            while True:
                i = 10
        raise unittest.SkipTest('https://github.com/apache/beam/issues/19461')

    def test_expand_kafka_read(self):
        if False:
            for i in range(10):
                print('nop')
        raise unittest.SkipTest('https://github.com/apache/beam/issues/19461')

    def test_expand_kafka_write(self):
        if False:
            for i in range(10):
                print('nop')
        raise unittest.SkipTest('https://github.com/apache/beam/issues/19461')

    def test_sql(self):
        if False:
            return 10
        raise unittest.SkipTest('https://github.com/apache/beam/issues/19461')

    def test_pack_combiners(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_pack_combiners(assert_using_counter_names=False)

class FlinkRunnerTestStreaming(FlinkRunnerTest):

    def create_options(self):
        if False:
            print('Hello World!')
        options = super().create_options()
        options.view_as(StandardOptions).streaming = True
        return options

    def test_callbacks_with_exception(self):
        if False:
            while True:
                i = 10
        self.enable_commit = True
        super().test_callbacks_with_exception()

    def test_register_finalizations(self):
        if False:
            i = 10
            return i + 15
        self.enable_commit = True
        super().test_register_finalizations()
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()