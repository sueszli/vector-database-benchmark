import copy
import itertools
import json
import logging
import shutil
import tempfile
import uuid
import zipfile
from concurrent import futures
from typing import TYPE_CHECKING
from typing import Dict
from typing import Iterator
from typing import Optional
from typing import Tuple
from typing import Union
import grpc
from google.protobuf import json_format
from google.protobuf import timestamp_pb2
from apache_beam.portability.api import beam_artifact_api_pb2_grpc
from apache_beam.portability.api import beam_job_api_pb2
from apache_beam.portability.api import beam_job_api_pb2_grpc
from apache_beam.portability.api import endpoints_pb2
from apache_beam.runners.portability import artifact_service
from apache_beam.utils.timestamp import Timestamp
if TYPE_CHECKING:
    from typing import BinaryIO
    from google.protobuf import struct_pb2
    from apache_beam.portability.api import beam_runner_api_pb2
_LOGGER = logging.getLogger(__name__)
StateEvent = Tuple[int, Union[timestamp_pb2.Timestamp, Timestamp]]

def make_state_event(state, timestamp):
    if False:
        i = 10
        return i + 15
    if isinstance(timestamp, Timestamp):
        proto_timestamp = timestamp.to_proto()
    elif isinstance(timestamp, timestamp_pb2.Timestamp):
        proto_timestamp = timestamp
    else:
        raise ValueError('Expected apache_beam.utils.timestamp.Timestamp, or google.protobuf.timestamp_pb2.Timestamp. Got %s' % type(timestamp))
    return beam_job_api_pb2.JobStateEvent(state=state, timestamp=proto_timestamp)

class AbstractJobServiceServicer(beam_job_api_pb2_grpc.JobServiceServicer):
    """Manages one or more pipelines, possibly concurrently.
  Experimental: No backward compatibility guaranteed.
  Servicer for the Beam Job API.
  """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._jobs = {}

    def create_beam_job(self, preparation_id, job_name, pipeline, options):
        if False:
            i = 10
            return i + 15
        'Returns an instance of AbstractBeamJob specific to this servicer.'
        raise NotImplementedError(type(self))

    def Prepare(self, request, context=None, timeout=None):
        if False:
            i = 10
            return i + 15
        _LOGGER.debug('Got Prepare request.')
        preparation_id = '%s-%s' % (request.job_name, uuid.uuid4())
        self._jobs[preparation_id] = self.create_beam_job(preparation_id, request.job_name, request.pipeline, request.pipeline_options)
        self._jobs[preparation_id].prepare()
        _LOGGER.debug("Prepared job '%s' as '%s'", request.job_name, preparation_id)
        return beam_job_api_pb2.PrepareJobResponse(preparation_id=preparation_id, artifact_staging_endpoint=self._jobs[preparation_id].artifact_staging_endpoint(), staging_session_token=preparation_id)

    def Run(self, request, context=None, timeout=None):
        if False:
            print('Hello World!')
        job_id = request.preparation_id
        _LOGGER.info("Running job '%s'", job_id)
        self._jobs[job_id].run()
        return beam_job_api_pb2.RunJobResponse(job_id=job_id)

    def GetJobs(self, request, context=None, timeout=None):
        if False:
            while True:
                i = 10
        return beam_job_api_pb2.GetJobsResponse(job_info=[job.to_runner_api() for job in self._jobs.values()])

    def GetState(self, request, context=None):
        if False:
            for i in range(10):
                print('nop')
        return make_state_event(*self._jobs[request.job_id].get_state())

    def GetPipeline(self, request, context=None, timeout=None):
        if False:
            while True:
                i = 10
        return beam_job_api_pb2.GetJobPipelineResponse(pipeline=self._jobs[request.job_id].get_pipeline())

    def Cancel(self, request, context=None, timeout=None):
        if False:
            return 10
        self._jobs[request.job_id].cancel()
        return beam_job_api_pb2.CancelJobResponse(state=self._jobs[request.job_id].get_state()[0])

    def GetStateStream(self, request, context=None, timeout=None):
        if False:
            i = 10
            return i + 15
        'Yields state transitions since the stream started.\n      '
        if request.job_id not in self._jobs:
            raise LookupError('Job {} does not exist'.format(request.job_id))
        job = self._jobs[request.job_id]
        for (state, timestamp) in job.get_state_stream():
            yield make_state_event(state, timestamp)

    def GetMessageStream(self, request, context=None, timeout=None):
        if False:
            return 10
        'Yields messages since the stream started.\n      '
        if request.job_id not in self._jobs:
            raise LookupError('Job {} does not exist'.format(request.job_id))
        job = self._jobs[request.job_id]
        for msg in job.get_message_stream():
            if isinstance(msg, tuple):
                resp = beam_job_api_pb2.JobMessagesResponse(state_response=make_state_event(*msg))
            else:
                resp = beam_job_api_pb2.JobMessagesResponse(message_response=msg)
            yield resp

    def DescribePipelineOptions(self, request, context=None, timeout=None):
        if False:
            print('Hello World!')
        return beam_job_api_pb2.DescribePipelineOptionsResponse()

class AbstractBeamJob(object):
    """Abstract baseclass for managing a single Beam job."""

    def __init__(self, job_id, job_name, pipeline, options):
        if False:
            print('Hello World!')
        self._job_id = job_id
        self._job_name = job_name
        self._pipeline_proto = pipeline
        self._pipeline_options = options
        self._state_history = [(beam_job_api_pb2.JobState.STOPPED, Timestamp.now())]

    def prepare(self):
        if False:
            while True:
                i = 10
        'Called immediately after this class is instantiated'
        raise NotImplementedError(self)

    def run(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError(self)

    def cancel(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError(self)

    def artifact_staging_endpoint(self):
        if False:
            return 10
        raise NotImplementedError(self)

    def get_state_stream(self):
        if False:
            print('Hello World!')
        raise NotImplementedError(self)

    def get_message_stream(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError(self)

    @property
    def state(self):
        if False:
            return 10
        'Get the latest state enum.'
        return self.get_state()[0]

    def get_state(self):
        if False:
            print('Hello World!')
        'Get a tuple of the latest state and its timestamp.'
        return self._state_history[-1]

    def set_state(self, new_state):
        if False:
            print('Hello World!')
        'Set the latest state as an int enum and update the state history.\n\n    :param new_state: int\n      latest state enum\n    :return: Timestamp or None\n      the new timestamp if the state has not changed, else None\n    '
        if new_state != self._state_history[-1][0]:
            timestamp = Timestamp.now()
            self._state_history.append((new_state, timestamp))
            return timestamp
        else:
            return None

    def with_state_history(self, state_stream):
        if False:
            for i in range(10):
                print('nop')
        'Utility to prepend recorded state history to an active state stream'
        return itertools.chain(self._state_history[:], state_stream)

    def get_pipeline(self):
        if False:
            for i in range(10):
                print('nop')
        return self._pipeline_proto

    @staticmethod
    def is_terminal_state(state):
        if False:
            while True:
                i = 10
        from apache_beam.runners.portability import portable_runner
        return state in portable_runner.TERMINAL_STATES

    def to_runner_api(self):
        if False:
            i = 10
            return i + 15
        return beam_job_api_pb2.JobInfo(job_id=self._job_id, job_name=self._job_name, pipeline_options=self._pipeline_options, state=self.state)

class JarArtifactManager(object):

    def __init__(self, jar_path, root):
        if False:
            i = 10
            return i + 15
        self._root = root
        self._zipfile_handle = zipfile.ZipFile(jar_path, 'a')

    def close(self):
        if False:
            while True:
                i = 10
        self._zipfile_handle.close()

    def file_writer(self, path):
        if False:
            return 10
        'Given a relative path, returns an open handle that can be written to\n    and an reference that can later be used to read this file.'
        full_path = '%s/%s' % (self._root, path)
        return (self._zipfile_handle.open(full_path, 'w', force_zip64=True), 'classpath://%s' % full_path)

    def zipfile_handle(self):
        if False:
            i = 10
            return i + 15
        return self._zipfile_handle

class UberJarBeamJob(AbstractBeamJob):
    """Abstract baseclass for creating a Beam job. The resulting job will be
  packaged and run in an executable uber jar."""
    PIPELINE_FOLDER = 'BEAM-PIPELINE'
    PIPELINE_MANIFEST = PIPELINE_FOLDER + '/pipeline-manifest.json'
    PIPELINE_NAME = 'pipeline'
    PIPELINE_PATH = '/'.join([PIPELINE_FOLDER, PIPELINE_NAME, 'pipeline.json'])
    PIPELINE_OPTIONS_PATH = '/'.join([PIPELINE_FOLDER, PIPELINE_NAME, 'pipeline-options.json'])
    ARTIFACT_FOLDER = '/'.join([PIPELINE_FOLDER, PIPELINE_NAME, 'artifacts'])

    def __init__(self, executable_jar, job_id, job_name, pipeline, options, artifact_port=0):
        if False:
            print('Hello World!')
        super().__init__(job_id, job_name, pipeline, options)
        self._executable_jar = executable_jar
        self._jar_uploaded = False
        self._artifact_port = artifact_port

    def prepare(self):
        if False:
            for i in range(10):
                print('nop')
        with tempfile.NamedTemporaryFile(suffix='.jar') as tout:
            self._jar = tout.name
        shutil.copy(self._executable_jar, self._jar)
        self._start_artifact_service(self._jar, self._artifact_port)

    def _start_artifact_service(self, jar, requested_port):
        if False:
            i = 10
            return i + 15
        self._artifact_manager = JarArtifactManager(self._jar, self.ARTIFACT_FOLDER)
        self._artifact_staging_service = artifact_service.ArtifactStagingService(self._artifact_manager.file_writer)
        self._artifact_staging_service.register_job(self._job_id, {env_id: env.dependencies for (env_id, env) in self._pipeline_proto.components.environments.items()})
        options = [('grpc.http2.max_pings_without_data', 0), ('grpc.http2.max_ping_strikes', 0)]
        self._artifact_staging_server = grpc.server(futures.ThreadPoolExecutor(), options=options)
        port = self._artifact_staging_server.add_insecure_port('[::]:%s' % requested_port)
        beam_artifact_api_pb2_grpc.add_ArtifactStagingServiceServicer_to_server(self._artifact_staging_service, self._artifact_staging_server)
        self._artifact_staging_endpoint = endpoints_pb2.ApiServiceDescriptor(url='localhost:%d' % port)
        self._artifact_staging_server.start()
        _LOGGER.info('Artifact server started on port %s', port)
        return port

    def _stop_artifact_service(self):
        if False:
            print('Hello World!')
        self._artifact_staging_server.stop(1)
        pipeline = copy.copy(self._pipeline_proto)
        if any((env.dependencies for env in pipeline.components.environments.values())):
            for (env_id, deps) in self._artifact_staging_service.resolved_deps(self._job_id).items():
                env = self._pipeline_proto.components.environments[env_id]
                del env.dependencies[:]
                env.dependencies.extend(deps)
        z = self._artifact_manager.zipfile_handle()
        with z.open(self.PIPELINE_PATH, 'w') as fout:
            fout.write(json_format.MessageToJson(self._pipeline_proto).encode('utf-8'))
        with z.open(self.PIPELINE_OPTIONS_PATH, 'w') as fout:
            fout.write(json_format.MessageToJson(self._pipeline_options).encode('utf-8'))
        with z.open(self.PIPELINE_MANIFEST, 'w') as fout:
            fout.write(json.dumps({'defaultJobName': self.PIPELINE_NAME}).encode('utf-8'))
        self._artifact_manager.close()

    def artifact_staging_endpoint(self):
        if False:
            print('Hello World!')
        return self._artifact_staging_endpoint