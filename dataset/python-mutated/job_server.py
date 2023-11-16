import atexit
import shutil
import signal
import tempfile
import threading
import grpc
from apache_beam.options import pipeline_options
from apache_beam.portability.api import beam_job_api_pb2_grpc
from apache_beam.runners.portability import local_job_service
from apache_beam.utils import subprocess_server
from apache_beam.version import __version__ as beam_version

class JobServer(object):

    def start(self):
        if False:
            for i in range(10):
                print('nop')
        'Starts this JobServer, returning a grpc service to which to submit jobs.\n    '
        raise NotImplementedError(type(self))

    def stop(self):
        if False:
            while True:
                i = 10
        'Stops this job server.'
        raise NotImplementedError(type(self))

class ExternalJobServer(JobServer):

    def __init__(self, endpoint, timeout=None):
        if False:
            while True:
                i = 10
        self._endpoint = endpoint
        self._timeout = timeout

    def start(self):
        if False:
            for i in range(10):
                print('nop')
        channel = grpc.insecure_channel(self._endpoint)
        grpc.channel_ready_future(channel).result(timeout=self._timeout)
        return beam_job_api_pb2_grpc.JobServiceStub(channel)

    def stop(self):
        if False:
            for i in range(10):
                print('nop')
        pass

class EmbeddedJobServer(JobServer):

    def start(self):
        if False:
            return 10
        return local_job_service.LocalJobServicer()

    def stop(self):
        if False:
            i = 10
            return i + 15
        pass

class StopOnExitJobServer(JobServer):
    """Wraps a JobServer such that its stop will automatically be called on exit.
  """

    def __init__(self, job_server):
        if False:
            for i in range(10):
                print('nop')
        self._lock = threading.Lock()
        self._job_server = job_server
        self._started = False

    def start(self):
        if False:
            print('Hello World!')
        with self._lock:
            if not self._started:
                self._endpoint = self._job_server.start()
                self._started = True
                atexit.register(self.stop)
                signal.signal(signal.SIGINT, self.stop)
        return self._endpoint

    def stop(self):
        if False:
            return 10
        with self._lock:
            if self._started:
                self._job_server.stop()
                self._started = False

class SubprocessJobServer(JobServer):
    """An abstract base class for JobServers run as an external process."""

    def __init__(self):
        if False:
            while True:
                i = 10
        self._local_temp_root = None
        self._server = None

    def subprocess_cmd_and_endpoint(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError(type(self))

    def start(self):
        if False:
            i = 10
            return i + 15
        if self._server is None:
            self._local_temp_root = tempfile.mkdtemp(prefix='beam-temp')
            (cmd, endpoint) = self.subprocess_cmd_and_endpoint()
            port = int(endpoint.split(':')[-1])
            self._server = subprocess_server.SubprocessServer(beam_job_api_pb2_grpc.JobServiceStub, cmd, port=port)
        return self._server.start()

    def stop(self):
        if False:
            print('Hello World!')
        if self._local_temp_root:
            shutil.rmtree(self._local_temp_root)
            self._local_temp_root = None
        return self._server.stop()

    def local_temp_dir(self, **kwargs):
        if False:
            print('Hello World!')
        return tempfile.mkdtemp(dir=self._local_temp_root, **kwargs)

class JavaJarJobServer(SubprocessJobServer):

    def __init__(self, options):
        if False:
            i = 10
            return i + 15
        super().__init__()
        options = options.view_as(pipeline_options.JobServerOptions)
        self._job_port = options.job_port
        self._artifact_port = options.artifact_port
        self._expansion_port = options.expansion_port
        self._artifacts_dir = options.artifacts_dir
        self._java_launcher = options.job_server_java_launcher
        self._jvm_properties = options.job_server_jvm_properties

    def java_arguments(self, job_port, artifact_port, expansion_port, artifacts_dir):
        if False:
            print('Hello World!')
        raise NotImplementedError(type(self))

    def path_to_jar(self):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError(type(self))

    @staticmethod
    def path_to_beam_jar(gradle_target, artifact_id=None):
        if False:
            for i in range(10):
                print('nop')
        return subprocess_server.JavaJarServer.path_to_beam_jar(gradle_target, artifact_id=artifact_id)

    @staticmethod
    def local_jar(url):
        if False:
            i = 10
            return i + 15
        return subprocess_server.JavaJarServer.local_jar(url)

    def subprocess_cmd_and_endpoint(self):
        if False:
            i = 10
            return i + 15
        jar_path = self.local_jar(self.path_to_jar())
        artifacts_dir = self._artifacts_dir if self._artifacts_dir else self.local_temp_dir(prefix='artifacts')
        (job_port,) = subprocess_server.pick_port(self._job_port)
        subprocess_cmd = [self._java_launcher, '-jar'] + self._jvm_properties + [jar_path] + list(self.java_arguments(job_port, self._artifact_port, self._expansion_port, artifacts_dir))
        return (subprocess_cmd, 'localhost:%s' % job_port)