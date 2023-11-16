import inspect
import logging
import socket
import subprocess
import sys
import time
import unittest
import grpc
import apache_beam as beam
from apache_beam.options.pipeline_options import DebugOptions
from apache_beam.options.pipeline_options import DirectOptions
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import PortableOptions
from apache_beam.portability import python_urns
from apache_beam.portability.api import beam_job_api_pb2
from apache_beam.portability.api import beam_job_api_pb2_grpc
from apache_beam.runners.portability import portable_runner
from apache_beam.runners.portability.fn_api_runner import fn_runner_test
from apache_beam.runners.portability.local_job_service import LocalJobServicer
from apache_beam.runners.portability.portable_runner import PortableRunner
from apache_beam.runners.worker import worker_pool_main
from apache_beam.runners.worker.channel_factory import GRPCChannelFactory
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to
from apache_beam.transforms import environments
from apache_beam.transforms import userstate
_LOGGER = logging.getLogger(__name__)

class PortableRunnerTest(fn_runner_test.FnApiRunnerTest):
    _use_subprocesses = False

    @classmethod
    def _pick_unused_port(cls):
        if False:
            for i in range(10):
                print('nop')
        return cls._pick_unused_ports(num_ports=1)[0]

    @staticmethod
    def _pick_unused_ports(num_ports):
        if False:
            i = 10
            return i + 15
        'Not perfect, but we have to provide a port to the subprocess.'
        sockets = []
        ports = []
        for _ in range(0, num_ports):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sockets.append(s)
            s.bind(('localhost', 0))
            (_, port) = s.getsockname()
            ports.append(port)
        try:
            return ports
        finally:
            for s in sockets:
                s.close()

    @classmethod
    def _start_local_runner_subprocess_job_service(cls):
        if False:
            return 10
        cls._maybe_kill_subprocess()
        (job_port, expansion_port) = cls._pick_unused_ports(num_ports=2)
        _LOGGER.info('Starting server on port %d.', job_port)
        cls._subprocess = subprocess.Popen(cls._subprocess_command(job_port, expansion_port))
        address = 'localhost:%d' % job_port
        job_service = beam_job_api_pb2_grpc.JobServiceStub(GRPCChannelFactory.insecure_channel(address))
        _LOGGER.info('Waiting for server to be ready...')
        start = time.time()
        timeout = 300
        while True:
            time.sleep(0.1)
            if cls._subprocess.poll() is not None:
                raise RuntimeError('Subprocess terminated unexpectedly with exit code %d.' % cls._subprocess.returncode)
            elif time.time() - start > timeout:
                raise RuntimeError('Pipeline timed out waiting for job service subprocess.')
            else:
                try:
                    job_service.GetState(beam_job_api_pb2.GetJobStateRequest(job_id='[fake]'))
                    break
                except grpc.RpcError as exn:
                    if exn.code() != grpc.StatusCode.UNAVAILABLE:
                        break
        _LOGGER.info('Server ready.')
        return address

    @classmethod
    def _get_job_endpoint(cls):
        if False:
            i = 10
            return i + 15
        if '_job_endpoint' not in cls.__dict__:
            cls._job_endpoint = cls._create_job_endpoint()
        return cls._job_endpoint

    @classmethod
    def _create_job_endpoint(cls):
        if False:
            return 10
        if cls._use_subprocesses:
            return cls._start_local_runner_subprocess_job_service()
        else:
            cls._servicer = LocalJobServicer()
            return 'localhost:%d' % cls._servicer.start_grpc_server()

    @classmethod
    def get_runner(cls):
        if False:
            for i in range(10):
                print('nop')
        return portable_runner.PortableRunner()

    @classmethod
    def tearDownClass(cls):
        if False:
            return 10
        cls._maybe_kill_subprocess()

    @classmethod
    def _maybe_kill_subprocess(cls):
        if False:
            return 10
        if hasattr(cls, '_subprocess') and cls._subprocess.poll() is None:
            cls._subprocess.kill()
            time.sleep(0.1)

    def create_options(self):
        if False:
            for i in range(10):
                print('nop')

        def get_pipeline_name():
            if False:
                return 10
            for (_, _, _, method_name, _, _) in inspect.stack():
                if method_name.find('test') != -1:
                    return method_name
            return 'unknown_test'
        options = PipelineOptions.from_dictionary({'job_name': get_pipeline_name() + '_' + str(time.time())})
        options.view_as(PortableOptions).job_endpoint = self._get_job_endpoint()
        options.view_as(PortableOptions).environment_type = python_urns.EMBEDDED_PYTHON
        options.view_as(DebugOptions).add_experiment('state_cache_size=100')
        options.view_as(DebugOptions).add_experiment('data_buffer_time_limit_ms=1000')
        return options

    def create_pipeline(self, is_drain=False):
        if False:
            return 10
        return beam.Pipeline(self.get_runner(), self.create_options())

    def test_pardo_state_with_custom_key_coder(self):
        if False:
            while True:
                i = 10
        "Tests that state requests work correctly when the key coder is an\n    SDK-specific coder, i.e. non standard coder. This is additionally enforced\n    by Java's ProcessBundleDescriptorsTest and by Flink's\n    ExecutableStageDoFnOperator which detects invalid encoding by checking for\n    the correct key group of the encoded key."
        index_state_spec = userstate.CombiningValueStateSpec('index', sum)
        n = 200
        duplicates = 1
        split = n // (duplicates + 1)
        inputs = [(i % split, str(i % split)) for i in range(0, n)]

        class Input(beam.DoFn):

            def process(self, impulse):
                if False:
                    i = 10
                    return i + 15
                for i in inputs:
                    yield i

        class AddIndex(beam.DoFn):

            def process(self, kv, index=beam.DoFn.StateParam(index_state_spec)):
                if False:
                    for i in range(10):
                        print('nop')
                (k, v) = kv
                index.add(1)
                yield (k, v, index.read())
        expected = [(i % split, str(i % split), i // split + 1) for i in range(0, n)]
        with self.create_pipeline() as p:
            assert_that(p | beam.Impulse() | beam.ParDo(Input()) | beam.ParDo(AddIndex()), equal_to(expected))

    def test_sdf_default_truncate_when_bounded(self):
        if False:
            i = 10
            return i + 15
        raise unittest.SkipTest("Portable runners don't support drain yet.")

    def test_sdf_default_truncate_when_unbounded(self):
        if False:
            return 10
        raise unittest.SkipTest("Portable runners don't support drain yet.")

    def test_sdf_with_truncate(self):
        if False:
            for i in range(10):
                print('nop')
        raise unittest.SkipTest("Portable runners don't support drain yet.")

    def test_draining_sdf_with_sdf_initiated_checkpointing(self):
        if False:
            while True:
                i = 10
        raise unittest.SkipTest("Portable runners don't support drain yet.")

@unittest.skip('https://github.com/apache/beam/issues/19422')
class PortableRunnerOptimized(PortableRunnerTest):

    def create_options(self):
        if False:
            return 10
        options = super().create_options()
        options.view_as(DebugOptions).add_experiment('pre_optimize=all')
        options.view_as(DebugOptions).add_experiment('state_cache_size=100')
        options.view_as(DebugOptions).add_experiment('data_buffer_time_limit_ms=1000')
        return options

class PortableRunnerOptimizedWithoutFusion(PortableRunnerTest):

    def create_options(self):
        if False:
            for i in range(10):
                print('nop')
        options = super().create_options()
        options.view_as(DebugOptions).add_experiment('pre_optimize=all_except_fusion')
        options.view_as(DebugOptions).add_experiment('state_cache_size=100')
        options.view_as(DebugOptions).add_experiment('data_buffer_time_limit_ms=1000')
        return options

class PortableRunnerTestWithExternalEnv(PortableRunnerTest):

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        (cls._worker_address, cls._worker_server) = worker_pool_main.BeamFnExternalWorkerPoolServicer.start(state_cache_size=100, data_buffer_time_limit_ms=1000)

    @classmethod
    def tearDownClass(cls):
        if False:
            while True:
                i = 10
        cls._worker_server.stop(1)

    def create_options(self):
        if False:
            for i in range(10):
                print('nop')
        options = super().create_options()
        options.view_as(PortableOptions).environment_type = 'EXTERNAL'
        options.view_as(PortableOptions).environment_config = self._worker_address
        return options

class PortableRunnerTestWithSubprocesses(PortableRunnerTest):
    _use_subprocesses = True

    def create_options(self):
        if False:
            i = 10
            return i + 15
        options = super().create_options()
        options.view_as(PortableOptions).environment_type = python_urns.SUBPROCESS_SDK
        options.view_as(PortableOptions).environment_config = (b'%s -m apache_beam.runners.worker.sdk_worker_main' % sys.executable.encode('ascii')).decode('utf-8')
        options.view_as(DebugOptions).add_experiment('state_cache_size=100')
        options.view_as(DebugOptions).add_experiment('data_buffer_time_limit_ms=1000')
        return options

    @classmethod
    def _subprocess_command(cls, job_port, _):
        if False:
            for i in range(10):
                print('nop')
        return [sys.executable, '-m', 'apache_beam.runners.portability.local_job_service_main', '-p', str(job_port)]

    def test_batch_rebatch_pardos(self):
        if False:
            print('Hello World!')
        raise unittest.SkipTest("Portable runners with subprocess can't make assertions about warnings raised on the worker.")

class PortableRunnerTestWithSubprocessesAndMultiWorkers(PortableRunnerTestWithSubprocesses):
    _use_subprocesses = True

    def create_options(self):
        if False:
            print('Hello World!')
        options = super().create_options()
        options.view_as(DirectOptions).direct_num_workers = 2
        return options

class PortableRunnerInternalTest(unittest.TestCase):

    def test__create_default_environment(self):
        if False:
            i = 10
            return i + 15
        docker_image = environments.DockerEnvironment.default_docker_image()
        self.assertEqual(PortableRunner._create_environment(PipelineOptions.from_dictionary({'sdk_location': 'container'})), environments.DockerEnvironment(container_image=docker_image))

    def test__create_docker_environment(self):
        if False:
            print('Hello World!')
        docker_image = 'py-docker'
        self.assertEqual(PortableRunner._create_environment(PipelineOptions.from_dictionary({'environment_type': 'DOCKER', 'environment_config': docker_image, 'sdk_location': 'container'})), environments.DockerEnvironment(container_image=docker_image))

    def test__create_process_environment(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(PortableRunner._create_environment(PipelineOptions.from_dictionary({'environment_type': 'PROCESS', 'environment_config': '{"os": "linux", "arch": "amd64", "command": "run.sh", "env":{"k1": "v1"} }', 'sdk_location': 'container'})), environments.ProcessEnvironment('run.sh', os='linux', arch='amd64', env={'k1': 'v1'}))
        self.assertEqual(PortableRunner._create_environment(PipelineOptions.from_dictionary({'environment_type': 'PROCESS', 'environment_config': '{"command": "run.sh"}', 'sdk_location': 'container'})), environments.ProcessEnvironment('run.sh'))

    def test__create_external_environment(self):
        if False:
            print('Hello World!')
        self.assertEqual(PortableRunner._create_environment(PipelineOptions.from_dictionary({'environment_type': 'EXTERNAL', 'environment_config': 'localhost:50000', 'sdk_location': 'container'})), environments.ExternalEnvironment('localhost:50000'))
        raw_config = ' {"url":"localhost:50000", "params":{"k1":"v1"}} '
        for env_config in (raw_config, raw_config.lstrip(), raw_config.strip()):
            self.assertEqual(PortableRunner._create_environment(PipelineOptions.from_dictionary({'environment_type': 'EXTERNAL', 'environment_config': env_config, 'sdk_location': 'container'})), environments.ExternalEnvironment('localhost:50000', params={'k1': 'v1'}))
        with self.assertRaises(ValueError):
            PortableRunner._create_environment(PipelineOptions.from_dictionary({'environment_type': 'EXTERNAL', 'environment_config': '{invalid}', 'sdk_location': 'container'}))
        with self.assertRaises(ValueError) as ctx:
            PortableRunner._create_environment(PipelineOptions.from_dictionary({'environment_type': 'EXTERNAL', 'environment_config': '{"params":{"k1":"v1"}}', 'sdk_location': 'container'}))
        self.assertIn('External environment endpoint must be set.', ctx.exception.args)

def hasDockerImage():
    if False:
        for i in range(10):
            print('nop')
    image = environments.DockerEnvironment.default_docker_image()
    try:
        check_image = subprocess.check_output('docker images -q %s' % image, shell=True)
        return check_image != ''
    except Exception:
        return False

@unittest.skipIf(not hasDockerImage(), 'docker not installed or no docker image')
class PortableRunnerTestWithLocalDocker(PortableRunnerTest):

    def create_options(self):
        if False:
            i = 10
            return i + 15
        options = super().create_options()
        options.view_as(PortableOptions).job_endpoint = 'embed'
        return options
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()