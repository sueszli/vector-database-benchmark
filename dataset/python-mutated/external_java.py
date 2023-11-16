"""Tests for the Java external transforms."""
import argparse
import logging
import subprocess
import sys
import grpc
from mock import patch
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to
from apache_beam.transforms.external import ImplicitSchemaPayloadBuilder
try:
    from apache_beam.runners.dataflow.internal import apiclient as _apiclient
except ImportError:
    apiclient = None
else:
    apiclient = _apiclient

class JavaExternalTransformTest(object):
    expansion_service_jar = None
    expansion_service_port = None

    class _RunWithExpansion(object):

        def __init__(self):
            if False:
                while True:
                    i = 10
            self._server = None

        def __enter__(self):
            if False:
                for i in range(10):
                    print('nop')
            if not (JavaExternalTransformTest.expansion_service_jar or JavaExternalTransformTest.expansion_service_port):
                raise RuntimeError('No expansion service jar or port provided.')
            JavaExternalTransformTest.expansion_service_port = JavaExternalTransformTest.expansion_service_port or 8091
            jar = JavaExternalTransformTest.expansion_service_jar
            port = JavaExternalTransformTest.expansion_service_port
            if jar:
                self._server = subprocess.Popen(['java', '-jar', jar, str(port)])
            address = 'localhost:%s' % str(port)
            with grpc.insecure_channel(address) as channel:
                grpc.channel_ready_future(channel).result()

        def __exit__(self, type, value, traceback):
            if False:
                print('Hello World!')
            if self._server:
                self._server.kill()
                self._server = None

    @staticmethod
    def test_java_expansion_dataflow():
        if False:
            for i in range(10):
                print('nop')
        if apiclient is None:
            return
        with patch.object(apiclient.DataflowApplicationClient, 'create_job') as mock_create_job:
            with JavaExternalTransformTest._RunWithExpansion():
                pipeline_options = PipelineOptions(['--runner=DataflowRunner', '--project=dummyproject', '--region=some-region1', '--experiments=beam_fn_api', '--temp_location=gs://dummybucket/'])
                JavaExternalTransformTest.run_pipeline(pipeline_options, JavaExternalTransformTest.expansion_service_port, False)
                mock_args = mock_create_job.call_args_list
                assert mock_args
                (args, kwargs) = mock_args[0]
                job = args[0]
                job_str = '%s' % job
                assert 'beam:transforms:xlang:filter_less_than_eq' in job_str

    @staticmethod
    def run_pipeline_with_expansion_service(pipeline_options):
        if False:
            while True:
                i = 10
        with JavaExternalTransformTest._RunWithExpansion():
            JavaExternalTransformTest.run_pipeline(pipeline_options, JavaExternalTransformTest.expansion_service_port, True)

    @staticmethod
    def run_pipeline(pipeline_options, expansion_service, wait_until_finish=True):
        if False:
            print('Hello World!')
        TEST_COUNT_URN = 'beam:transforms:xlang:count'
        TEST_FILTER_URN = 'beam:transforms:xlang:filter_less_than_eq'
        p = TestPipeline(options=pipeline_options)
        if isinstance(expansion_service, int):
            expansion_service = 'localhost:%s' % str(expansion_service)
        res = p | beam.Create(list('aaabccxyyzzz')) | beam.Map(str) | beam.ExternalTransform(TEST_FILTER_URN, ImplicitSchemaPayloadBuilder({'data': 'middle'}), expansion_service) | beam.ExternalTransform(TEST_COUNT_URN, None, expansion_service) | beam.Map(lambda kv: '%s: %s' % kv)
        assert_that(res, equal_to(['a: 3', 'b: 1', 'c: 2']))
        result = p.run()
        if wait_until_finish:
            result.wait_until_finish()
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--expansion_service_jar')
    parser.add_argument('--expansion_service_port')
    parser.add_argument('--expansion_service_target')
    parser.add_argument('--expansion_service_target_appendix')
    (known_args, pipeline_args) = parser.parse_known_args(sys.argv)
    if known_args.expansion_service_jar:
        JavaExternalTransformTest.expansion_service_jar = known_args.expansion_service_jar
        JavaExternalTransformTest.expansion_service_port = int(known_args.expansion_service_port)
        pipeline_options = PipelineOptions(pipeline_args)
        JavaExternalTransformTest.run_pipeline_with_expansion_service(pipeline_options)
    elif known_args.expansion_service_target:
        pipeline_options = PipelineOptions(pipeline_args)
        JavaExternalTransformTest.run_pipeline(pipeline_options, beam.transforms.external.BeamJarExpansionService(known_args.expansion_service_target, gradle_appendix=known_args.expansion_service_target_appendix))
    else:
        raise RuntimeError('--expansion_service_jar or --expansion_service_target should be provided.')