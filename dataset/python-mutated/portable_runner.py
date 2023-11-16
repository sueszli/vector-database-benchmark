import atexit
import copy
import functools
import itertools
import logging
import threading
import time
from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import Iterator
from typing import Optional
from typing import Tuple
import grpc
from apache_beam.metrics import metric
from apache_beam.metrics.execution import MetricResult
from apache_beam.options.pipeline_options import DebugOptions
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import PortableOptions
from apache_beam.options.pipeline_options import StandardOptions
from apache_beam.options.value_provider import ValueProvider
from apache_beam.portability import common_urns
from apache_beam.portability import python_urns
from apache_beam.portability.api import beam_artifact_api_pb2_grpc
from apache_beam.portability.api import beam_job_api_pb2
from apache_beam.portability.api import beam_runner_api_pb2
from apache_beam.runners import runner
from apache_beam.runners.job import utils as job_utils
from apache_beam.runners.portability import artifact_service
from apache_beam.runners.portability import job_server
from apache_beam.runners.portability import portable_metrics
from apache_beam.runners.portability.fn_api_runner.fn_runner import translations
from apache_beam.runners.worker import sdk_worker_main
from apache_beam.runners.worker import worker_pool_main
from apache_beam.transforms import environments
if TYPE_CHECKING:
    from google.protobuf import struct_pb2
    from apache_beam.pipeline import Pipeline
__all__ = ['PortableRunner']
MESSAGE_LOG_LEVELS = {beam_job_api_pb2.JobMessage.MESSAGE_IMPORTANCE_UNSPECIFIED: logging.INFO, beam_job_api_pb2.JobMessage.JOB_MESSAGE_DEBUG: logging.DEBUG, beam_job_api_pb2.JobMessage.JOB_MESSAGE_DETAILED: logging.DEBUG, beam_job_api_pb2.JobMessage.JOB_MESSAGE_BASIC: logging.INFO, beam_job_api_pb2.JobMessage.JOB_MESSAGE_WARNING: logging.WARNING, beam_job_api_pb2.JobMessage.JOB_MESSAGE_ERROR: logging.ERROR}
TERMINAL_STATES = [beam_job_api_pb2.JobState.DONE, beam_job_api_pb2.JobState.DRAINED, beam_job_api_pb2.JobState.FAILED, beam_job_api_pb2.JobState.CANCELLED]
_LOGGER = logging.getLogger(__name__)

class JobServiceHandle(object):
    """
  Encapsulates the interactions necessary to submit a pipeline to a job service.

  The base set of interactions consists of 3 steps:
  - prepare
  - stage
  - run
  """

    def __init__(self, job_service, options, retain_unknown_options=False):
        if False:
            print('Hello World!')
        self.job_service = job_service
        self.options = options
        self.timeout = options.view_as(PortableOptions).job_server_timeout
        self.artifact_endpoint = options.view_as(PortableOptions).artifact_endpoint
        self._retain_unknown_options = retain_unknown_options

    def submit(self, proto_pipeline):
        if False:
            while True:
                i = 10
        '\n    Submit and run the pipeline defined by `proto_pipeline`.\n    '
        prepare_response = self.prepare(proto_pipeline)
        artifact_endpoint = self.artifact_endpoint or prepare_response.artifact_staging_endpoint.url
        self.stage(proto_pipeline, artifact_endpoint, prepare_response.staging_session_token)
        return self.run(prepare_response.preparation_id)

    def get_pipeline_options(self):
        if False:
            print('Hello World!')
        '\n    Get `self.options` as a protobuf Struct\n    '

        def send_options_request(max_retries=5):
            if False:
                i = 10
                return i + 15
            num_retries = 0
            while True:
                try:
                    return self.job_service.DescribePipelineOptions(beam_job_api_pb2.DescribePipelineOptionsRequest(), timeout=self.timeout)
                except grpc.FutureTimeoutError:
                    raise
                except grpc.RpcError as e:
                    num_retries += 1
                    if num_retries > max_retries:
                        raise e
                    time.sleep(1)
        options_response = send_options_request()

        def add_runner_options(parser):
            if False:
                return 10
            for option in options_response.options:
                try:
                    add_arg_args = {'action': 'store', 'help': option.description}
                    if option.type == beam_job_api_pb2.PipelineOptionType.BOOLEAN:
                        add_arg_args['action'] = 'store_true' if option.default_value != 'true' else 'store_false'
                    elif option.type == beam_job_api_pb2.PipelineOptionType.INTEGER:
                        add_arg_args['type'] = int
                    elif option.type == beam_job_api_pb2.PipelineOptionType.ARRAY:
                        add_arg_args['action'] = 'append'
                    parser.add_argument('--%s' % option.name, **add_arg_args)
                except Exception as e:
                    if 'conflicting option string' not in str(e):
                        raise
                    _LOGGER.debug("Runner option '%s' was already added" % option.name)
        all_options = self.options.get_all_options(add_extra_args_fn=add_runner_options, retain_unknown_options=self._retain_unknown_options)
        return self.encode_pipeline_options(all_options)

    @staticmethod
    def encode_pipeline_options(all_options: Dict[str, Any]) -> 'struct_pb2.Struct':
        if False:
            while True:
                i = 10

        def convert_pipeline_option_value(v):
            if False:
                while True:
                    i = 10
            if type(v) == int:
                return str(v)
            elif isinstance(v, ValueProvider):
                return convert_pipeline_option_value(v.get()) if v.is_accessible() else None
            return v
        p_options = {'beam:option:' + k + ':v1': convert_pipeline_option_value(v) for (k, v) in all_options.items() if v is not None}
        return job_utils.dict_to_struct(p_options)

    def prepare(self, proto_pipeline):
        if False:
            return 10
        'Prepare the job on the job service'
        return self.job_service.Prepare(beam_job_api_pb2.PrepareJobRequest(job_name='job', pipeline=proto_pipeline, pipeline_options=self.get_pipeline_options()), timeout=self.timeout)

    def stage(self, proto_pipeline, artifact_staging_endpoint, staging_session_token):
        if False:
            return 10
        'Stage artifacts'
        if artifact_staging_endpoint:
            artifact_service.offer_artifacts(beam_artifact_api_pb2_grpc.ArtifactStagingServiceStub(channel=grpc.insecure_channel(artifact_staging_endpoint)), artifact_service.ArtifactRetrievalService(artifact_service.BeamFilesystemHandler(None).file_reader), staging_session_token)

    def run(self, preparation_id):
        if False:
            while True:
                i = 10
        'Run the job'
        try:
            state_stream = self.job_service.GetStateStream(beam_job_api_pb2.GetJobStateRequest(job_id=preparation_id), timeout=self.timeout)
            state_stream = itertools.chain([next(state_stream)], state_stream)
            message_stream = self.job_service.GetMessageStream(beam_job_api_pb2.JobMessagesRequest(job_id=preparation_id), timeout=self.timeout)
        except Exception:
            state_stream = message_stream = None
        run_response = self.job_service.Run(beam_job_api_pb2.RunJobRequest(preparation_id=preparation_id))
        if state_stream is None:
            state_stream = self.job_service.GetStateStream(beam_job_api_pb2.GetJobStateRequest(job_id=run_response.job_id))
            message_stream = self.job_service.GetMessageStream(beam_job_api_pb2.JobMessagesRequest(job_id=run_response.job_id))
        return (run_response.job_id, message_stream, state_stream)

class PortableRunner(runner.PipelineRunner):
    """
    Experimental: No backward compatibility guaranteed.
    A BeamRunner that executes Python pipelines via the Beam Job API.

    This runner is a stub and does not run the actual job.
    This runner schedules the job on a job service. The responsibility of
    running and managing the job lies with the job service used.
  """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._dockerized_job_server = None

    @staticmethod
    def _create_environment(options):
        if False:
            return 10
        return environments.Environment.from_options(options.view_as(PortableOptions))

    def default_job_server(self, options):
        if False:
            print('Hello World!')
        raise NotImplementedError('You must specify a --job_endpoint when using --runner=PortableRunner. Alternatively, you may specify which portable runner you intend to use, such as --runner=FlinkRunner or --runner=SparkRunner.')

    def create_job_service_handle(self, job_service, options):
        if False:
            for i in range(10):
                print('nop')
        return JobServiceHandle(job_service, options)

    def create_job_service(self, options):
        if False:
            i = 10
            return i + 15
        '\n    Start the job service and return a `JobServiceHandle`\n    '
        job_endpoint = options.view_as(PortableOptions).job_endpoint
        if job_endpoint:
            if job_endpoint == 'embed':
                server = job_server.EmbeddedJobServer()
            else:
                job_server_timeout = options.view_as(PortableOptions).job_server_timeout
                server = job_server.ExternalJobServer(job_endpoint, job_server_timeout)
        else:
            server = self.default_job_server(options)
        return self.create_job_service_handle(server.start(), options)

    @staticmethod
    def get_proto_pipeline(pipeline, options):
        if False:
            for i in range(10):
                print('nop')
        proto_pipeline = pipeline.to_runner_api(default_environment=environments.Environment.from_options(options.view_as(PortableOptions)))
        return PortableRunner._optimize_pipeline(proto_pipeline, options)

    @staticmethod
    def _optimize_pipeline(proto_pipeline: beam_runner_api_pb2.Pipeline, options: PipelineOptions) -> beam_runner_api_pb2.Pipeline:
        if False:
            return 10
        pre_optimize = options.view_as(DebugOptions).lookup_experiment('pre_optimize', 'default').lower()
        if not options.view_as(StandardOptions).streaming and pre_optimize != 'none':
            if pre_optimize == 'default':
                phases = [translations.pack_combiners, translations.lift_combiners, translations.sort_stages]
                partial = True
            elif pre_optimize == 'all':
                phases = translations.standard_optimize_phases()
                partial = False
            elif pre_optimize == 'all_except_fusion':
                phases = translations.standard_optimize_phases()
                phases.remove(translations.greedily_fuse)
                partial = True
            else:
                phases = []
                for phase_name in pre_optimize.split(','):
                    if phase_name in ('pack_combiners', 'lift_combiners'):
                        phases.append(getattr(translations, phase_name))
                    else:
                        raise ValueError('Unknown or inapplicable phase for pre_optimize: %s' % phase_name)
                phases.append(translations.sort_stages)
                partial = True
            known_urns = frozenset([common_urns.composites.RESHUFFLE.urn, common_urns.primitives.IMPULSE.urn, common_urns.primitives.FLATTEN.urn, common_urns.primitives.GROUP_BY_KEY.urn])
            proto_pipeline = translations.optimize_pipeline(proto_pipeline, phases=phases, known_runner_urns=known_urns, partial=partial)
        return proto_pipeline

    def run_portable_pipeline(self, pipeline: beam_runner_api_pb2.Pipeline, options: PipelineOptions) -> runner.PipelineResult:
        if False:
            i = 10
            return i + 15
        portable_options = options.view_as(PortableOptions)
        portable_options.view_as(StandardOptions).runner = None
        cleanup_callbacks = self.start_and_replace_loopback_environments(pipeline, options)
        optimized_pipeline = self._optimize_pipeline(pipeline, options)
        job_service_handle = self.create_job_service(options)
        (job_id, message_stream, state_stream) = job_service_handle.submit(optimized_pipeline)
        result = PipelineResult(job_service_handle.job_service, job_id, message_stream, state_stream, cleanup_callbacks)
        if cleanup_callbacks:
            atexit.register(functools.partial(result._cleanup, on_exit=True))
            _LOGGER.info('Environment "%s" has started a component necessary for the execution. Be sure to run the pipeline using\n  with Pipeline() as p:\n    p.apply(..)\nThis ensures that the pipeline finishes before this program exits.', portable_options.environment_type)
        return result

    @staticmethod
    def start_and_replace_loopback_environments(pipeline, options):
        if False:
            return 10
        portable_options = copy.deepcopy(options.view_as(PortableOptions))
        experiments = options.view_as(DebugOptions).experiments or []
        cleanup_callbacks = []
        for env in pipeline.components.environments.values():
            if env.urn == python_urns.EMBEDDED_PYTHON_LOOPBACK:
                use_loopback_process_worker = options.view_as(DebugOptions).lookup_experiment('use_loopback_process_worker', False)
                portable_options.environment_type = 'EXTERNAL'
                (portable_options.environment_config, server) = worker_pool_main.BeamFnExternalWorkerPoolServicer.start(state_cache_size=sdk_worker_main._get_state_cache_size_bytes(options=options), data_buffer_time_limit_ms=sdk_worker_main._get_data_buffer_time_limit_ms(experiments), use_process=use_loopback_process_worker)
                external_env = environments.ExternalEnvironment.from_options(portable_options).to_runner_api(None)
                env.urn = external_env.urn
                env.payload = external_env.payload
                cleanup_callbacks.append(functools.partial(server.stop, 1))
        return cleanup_callbacks

class PortableMetrics(metric.MetricResults):

    def __init__(self, job_metrics_response):
        if False:
            while True:
                i = 10
        metrics = job_metrics_response.metrics
        self.attempted = portable_metrics.from_monitoring_infos(metrics.attempted)
        self.committed = portable_metrics.from_monitoring_infos(metrics.committed)

    @staticmethod
    def _combine(committed, attempted, filter):
        if False:
            print('Hello World!')
        all_keys = set(committed.keys()) | set(attempted.keys())
        return [MetricResult(key, committed.get(key), attempted.get(key)) for key in all_keys if metric.MetricResults.matches(filter, key)]

    def query(self, filter=None):
        if False:
            while True:
                i = 10
        (counters, distributions, gauges) = [self._combine(x, y, filter) for (x, y) in zip(self.committed, self.attempted)]
        return {self.COUNTERS: counters, self.DISTRIBUTIONS: distributions, self.GAUGES: gauges}

class PipelineResult(runner.PipelineResult):

    def __init__(self, job_service, job_id, message_stream, state_stream, cleanup_callbacks=()):
        if False:
            i = 10
            return i + 15
        super().__init__(beam_job_api_pb2.JobState.UNSPECIFIED)
        self._job_service = job_service
        self._job_id = job_id
        self._messages = []
        self._message_stream = message_stream
        self._state_stream = state_stream
        self._cleanup_callbacks = cleanup_callbacks
        self._metrics = None
        self._runtime_exception = None

    def cancel(self):
        if False:
            print('Hello World!')
        try:
            self._job_service.Cancel(beam_job_api_pb2.CancelJobRequest(job_id=self._job_id))
        finally:
            self._cleanup()

    @property
    def state(self):
        if False:
            return 10
        runner_api_state = self._job_service.GetState(beam_job_api_pb2.GetJobStateRequest(job_id=self._job_id)).state
        self._state = self.runner_api_state_to_pipeline_state(runner_api_state)
        return self._state

    @staticmethod
    def runner_api_state_to_pipeline_state(runner_api_state):
        if False:
            while True:
                i = 10
        return getattr(runner.PipelineState, beam_job_api_pb2.JobState.Enum.Name(runner_api_state))

    @staticmethod
    def pipeline_state_to_runner_api_state(pipeline_state):
        if False:
            for i in range(10):
                print('nop')
        if pipeline_state == runner.PipelineState.PENDING:
            return beam_job_api_pb2.JobState.STARTING
        else:
            try:
                return beam_job_api_pb2.JobState.Enum.Value(pipeline_state)
            except ValueError:
                return beam_job_api_pb2.JobState.UNSPECIFIED

    def metrics(self):
        if False:
            for i in range(10):
                print('nop')
        if not self._metrics:
            job_metrics_response = self._job_service.GetJobMetrics(beam_job_api_pb2.GetJobMetricsRequest(job_id=self._job_id))
            self._metrics = PortableMetrics(job_metrics_response)
        return self._metrics

    def _last_error_message(self):
        if False:
            for i in range(10):
                print('nop')
        messages = [m.message_response for m in self._messages if m.HasField('message_response')]
        error_messages = [m for m in messages if m.importance == beam_job_api_pb2.JobMessage.JOB_MESSAGE_ERROR]
        if error_messages:
            return error_messages[-1].message_text
        else:
            return 'unknown error'

    def wait_until_finish(self, duration=None):
        if False:
            i = 10
            return i + 15
        '\n    :param duration: The maximum time in milliseconds to wait for the result of\n    the execution. If None or zero, will wait until the pipeline finishes.\n    :return: The result of the pipeline, i.e. PipelineResult.\n    '

        def read_messages():
            if False:
                return 10
            previous_state = -1
            for message in self._message_stream:
                if message.HasField('message_response'):
                    logging.log(MESSAGE_LOG_LEVELS[message.message_response.importance], '%s', message.message_response.message_text)
                else:
                    current_state = message.state_response.state
                    if current_state != previous_state:
                        _LOGGER.info('Job state changed to %s', self.runner_api_state_to_pipeline_state(current_state))
                        previous_state = current_state
                self._messages.append(message)
        message_thread = threading.Thread(target=read_messages, name='wait_until_finish_read')
        message_thread.daemon = True
        message_thread.start()
        if duration:
            state_thread = threading.Thread(target=functools.partial(self._observe_state, message_thread), name='wait_until_finish_state_observer')
            state_thread.daemon = True
            state_thread.start()
            start_time = time.time()
            duration_secs = duration / 1000
            while time.time() - start_time < duration_secs and state_thread.is_alive():
                time.sleep(1)
        else:
            self._observe_state(message_thread)
        if self._runtime_exception:
            raise self._runtime_exception
        return self._state

    def _observe_state(self, message_thread):
        if False:
            for i in range(10):
                print('nop')
        try:
            for state_response in self._state_stream:
                self._state = self.runner_api_state_to_pipeline_state(state_response.state)
                if state_response.state in TERMINAL_STATES:
                    message_thread.join(10)
                    break
            if self._state != runner.PipelineState.DONE:
                self._runtime_exception = RuntimeError('Pipeline %s failed in state %s: %s' % (self._job_id, self._state, self._last_error_message()))
        except Exception as e:
            self._runtime_exception = e
        finally:
            self._cleanup()

    def _cleanup(self, on_exit=False):
        if False:
            print('Hello World!')
        if on_exit and self._cleanup_callbacks:
            _LOGGER.info('Running cleanup on exit. If your pipeline should continue running, be sure to use the following syntax:\n  with Pipeline() as p:\n    p.apply(..)\nThis ensures that the pipeline finishes before this program exits.')
        callback_exceptions = []
        for callback in self._cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                callback_exceptions.append(e)
        self._cleanup_callbacks = ()
        if callback_exceptions:
            formatted_exceptions = ''.join([f'\n\t{repr(e)}' for e in callback_exceptions])
            raise RuntimeError('Errors: {}'.format(formatted_exceptions))