import argparse
import logging
import shlex
import unittest
from shutil import rmtree
from tempfile import mkdtemp
import pytest
from apache_beam.options.pipeline_options import PortableOptions
from apache_beam.runners.portability import job_server
from apache_beam.runners.portability import portable_runner
from apache_beam.runners.portability import portable_runner_test
_LOGGER = logging.getLogger(__name__)

class SparkRunnerTest(portable_runner_test.PortableRunnerTest):
    _use_grpc = True
    _use_subprocesses = True
    expansion_port = None
    spark_job_server_jar = None

    @pytest.fixture(autouse=True)
    def parse_options(self, request):
        if False:
            i = 10
            return i + 15
        if not request.config.option.test_pipeline_options:
            raise unittest.SkipTest('Skipping because --test-pipeline-options is not specified.')
        test_pipeline_options = request.config.option.test_pipeline_options
        parser = argparse.ArgumentParser(add_help=True)
        parser.add_argument('--spark_job_server_jar', help='Job server jar to submit jobs.', action='store')
        parser.add_argument('--environment_type', default='LOOPBACK', choices=['DOCKER', 'PROCESS', 'LOOPBACK'], help='Set the environment type for running user code. DOCKER runs user code in a container. PROCESS runs user code in automatically started processes. LOOPBACK runs user code on the same process that originally submitted the job.')
        parser.add_argument('--environment_option', '--environment_options', dest='environment_options', action='append', default=None, help='Environment configuration for running the user code. Recognized options depend on --environment_type.\n For DOCKER: docker_container_image (optional)\n For PROCESS: process_command (required), process_variables (optional, comma-separated)\n For EXTERNAL: external_service_address (required)')
        (known_args, unknown_args) = parser.parse_known_args(shlex.split(test_pipeline_options))
        if unknown_args:
            _LOGGER.warning('Discarding unrecognized arguments %s' % unknown_args)
        self.set_spark_job_server_jar(known_args.spark_job_server_jar or job_server.JavaJarJobServer.path_to_beam_jar(':runners:spark:3:job-server:shadowJar'))
        self.environment_type = known_args.environment_type
        self.environment_options = known_args.environment_options

    @classmethod
    def _subprocess_command(cls, job_port, expansion_port):
        if False:
            while True:
                i = 10
        tmp_dir = mkdtemp(prefix='sparktest')
        cls.expansion_port = expansion_port
        try:
            return ['java', '-Dbeam.spark.test.reuseSparkContext=true', '-jar', cls.spark_job_server_jar, '--spark-master-url', 'local', '--artifacts-dir', tmp_dir, '--job-port', str(job_port), '--artifact-port', '0', '--expansion-port', str(expansion_port)]
        finally:
            rmtree(tmp_dir)

    @classmethod
    def get_runner(cls):
        if False:
            return 10
        return portable_runner.PortableRunner()

    @classmethod
    def get_expansion_service(cls):
        if False:
            print('Hello World!')
        return 'localhost:%s' % cls.expansion_port

    @classmethod
    def set_spark_job_server_jar(cls, spark_job_server_jar):
        if False:
            print('Hello World!')
        cls.spark_job_server_jar = spark_job_server_jar

    def create_options(self):
        if False:
            return 10
        options = super().create_options()
        options.view_as(PortableOptions).environment_type = self.environment_type
        options.view_as(PortableOptions).environment_options = self.environment_options
        return options

    def test_metrics(self):
        if False:
            print('Hello World!')
        raise unittest.SkipTest('https://github.com/apache/beam/issues/19496')

    def test_sdf(self):
        if False:
            for i in range(10):
                print('nop')
        raise unittest.SkipTest('https://github.com/apache/beam/issues/19468')

    def test_sdf_with_watermark_tracking(self):
        if False:
            i = 10
            return i + 15
        raise unittest.SkipTest('https://github.com/apache/beam/issues/19468')

    def test_sdf_with_sdf_initiated_checkpointing(self):
        if False:
            i = 10
            return i + 15
        raise unittest.SkipTest('https://github.com/apache/beam/issues/19468')

    def test_sdf_synthetic_source(self):
        if False:
            print('Hello World!')
        raise unittest.SkipTest('https://github.com/apache/beam/issues/19468')

    def test_callbacks_with_exception(self):
        if False:
            i = 10
            return i + 15
        raise unittest.SkipTest('https://github.com/apache/beam/issues/19517')

    def test_register_finalizations(self):
        if False:
            print('Hello World!')
        raise unittest.SkipTest('https://github.com/apache/beam/issues/19517')

    def test_sdf_with_dofn_as_watermark_estimator(self):
        if False:
            return 10
        raise unittest.SkipTest('https://github.com/apache/beam/issues/19468')

    def test_pardo_dynamic_timer(self):
        if False:
            print('Hello World!')
        raise unittest.SkipTest('https://github.com/apache/beam/issues/20179')

    def test_flattened_side_input(self):
        if False:
            for i in range(10):
                print('nop')
        super().test_flattened_side_input(with_transcoding=False)

    def test_custom_merging_window(self):
        if False:
            for i in range(10):
                print('nop')
        raise unittest.SkipTest('https://github.com/apache/beam/issues/20641')
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()