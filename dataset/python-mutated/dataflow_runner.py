"""A runner implementation that submits a job for remote execution.

The runner will create a JSON description of the job graph and then submit it
to the Dataflow Service for remote execution by a worker.
"""
import logging
import os
import threading
import time
import warnings
from collections import defaultdict
from subprocess import DEVNULL
from typing import TYPE_CHECKING
from typing import List
import apache_beam as beam
from apache_beam import coders
from apache_beam.options.pipeline_options import DebugOptions
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.options.pipeline_options import StandardOptions
from apache_beam.options.pipeline_options import TestOptions
from apache_beam.options.pipeline_options import TypeOptions
from apache_beam.options.pipeline_options import WorkerOptions
from apache_beam.portability import common_urns
from apache_beam.runners.common import group_by_key_input_visitor
from apache_beam.runners.dataflow.internal.clients import dataflow as dataflow_api
from apache_beam.runners.runner import PipelineResult
from apache_beam.runners.runner import PipelineRunner
from apache_beam.runners.runner import PipelineState
from apache_beam.typehints import typehints
from apache_beam.utils import processes
from apache_beam.utils.interactive_utils import is_in_notebook
from apache_beam.utils.plugin import BeamPlugin
if TYPE_CHECKING:
    from apache_beam.pipeline import PTransformOverride
__all__ = ['DataflowRunner']
_LOGGER = logging.getLogger(__name__)
BQ_SOURCE_UW_ERROR = 'The Read(BigQuerySource(...)) transform is not supported with newer stack features (Fn API, Dataflow Runner V2, etc). Please use the transform apache_beam.io.gcp.bigquery.ReadFromBigQuery instead.'

class DataflowRunner(PipelineRunner):
    """A runner that creates job graphs and submits them for remote execution.

  Every execution of the run() method will submit an independent job for
  remote execution that consists of the nodes reachable from the passed-in
  node argument or entire graph if the node is None. The run() method returns
  after the service creates the job, and the job status is reported as RUNNING.
  """
    from apache_beam.runners.dataflow.ptransform_overrides import NativeReadPTransformOverride
    _PTRANSFORM_OVERRIDES = [NativeReadPTransformOverride()]

    def __init__(self, cache=None):
        if False:
            print('Hello World!')
        self._default_environment = None

    def is_fnapi_compatible(self):
        if False:
            return 10
        return False

    @staticmethod
    def poll_for_job_completion(runner, result, duration, state_update_callback=None):
        if False:
            return 10
        'Polls for the specified job to finish running (successfully or not).\n\n    Updates the result with the new job information before returning.\n\n    Args:\n      runner: DataflowRunner instance to use for polling job state.\n      result: DataflowPipelineResult instance used for job information.\n      duration (int): The time to wait (in milliseconds) for job to finish.\n        If it is set to :data:`None`, it will wait indefinitely until the job\n        is finished.\n    '
        if result.state == PipelineState.DONE:
            return
        last_message_time = None
        current_seen_messages = set()
        last_error_rank = float('-inf')
        last_error_msg = None
        last_job_state = None
        final_countdown_timer_secs = 50.0
        sleep_secs = 5.0

        def rank_error(msg):
            if False:
                return 10
            if 'work item was attempted' in msg:
                return -1
            elif 'Traceback' in msg:
                return 1
            return 0
        if duration:
            start_secs = time.time()
            duration_secs = duration // 1000
        job_id = result.job_id()
        while True:
            response = runner.dataflow_client.get_job(job_id)
            if response.currentState is not None:
                if response.currentState != last_job_state:
                    if state_update_callback:
                        state_update_callback(response.currentState)
                    _LOGGER.info('Job %s is in state %s', job_id, response.currentState)
                    last_job_state = response.currentState
                if str(response.currentState) != 'JOB_STATE_RUNNING':
                    if final_countdown_timer_secs <= 0.0 or last_error_msg is not None or str(response.currentState) == 'JOB_STATE_DONE' or (str(response.currentState) == 'JOB_STATE_CANCELLED') or (str(response.currentState) == 'JOB_STATE_UPDATED') or (str(response.currentState) == 'JOB_STATE_DRAINED'):
                        break
                    if str(response.currentState) not in ('JOB_STATE_PENDING', 'JOB_STATE_QUEUED'):
                        sleep_secs = 1.0
                        final_countdown_timer_secs -= sleep_secs
            time.sleep(sleep_secs)
            page_token = None
            while True:
                (messages, page_token) = runner.dataflow_client.list_messages(job_id, page_token=page_token, start_time=last_message_time)
                for m in messages:
                    message = '%s: %s: %s' % (m.time, m.messageImportance, m.messageText)
                    if not last_message_time or m.time > last_message_time:
                        last_message_time = m.time
                        current_seen_messages = set()
                    if message in current_seen_messages:
                        continue
                    else:
                        current_seen_messages.add(message)
                    if m.messageImportance is None:
                        continue
                    message_importance = str(m.messageImportance)
                    if message_importance == 'JOB_MESSAGE_DEBUG' or message_importance == 'JOB_MESSAGE_DETAILED':
                        _LOGGER.debug(message)
                    elif message_importance == 'JOB_MESSAGE_BASIC':
                        _LOGGER.info(message)
                    elif message_importance == 'JOB_MESSAGE_WARNING':
                        _LOGGER.warning(message)
                    elif message_importance == 'JOB_MESSAGE_ERROR':
                        _LOGGER.error(message)
                        if rank_error(m.messageText) >= last_error_rank:
                            last_error_rank = rank_error(m.messageText)
                            last_error_msg = m.messageText
                    else:
                        _LOGGER.info(message)
                if not page_token:
                    break
            if duration:
                passed_secs = time.time() - start_secs
                if passed_secs > duration_secs:
                    _LOGGER.warning('Timing out on waiting for job %s after %d seconds', job_id, passed_secs)
                    break
        result._job = response
        runner.last_error_msg = last_error_msg

    @staticmethod
    def _only_element(iterable):
        if False:
            return 10
        (element,) = iterable
        return element

    @staticmethod
    def side_input_visitor(deterministic_key_coders=True):
        if False:
            print('Hello World!')
        from apache_beam.pipeline import PipelineVisitor
        from apache_beam.transforms.core import ParDo

        class SideInputVisitor(PipelineVisitor):
            """Ensures input `PCollection` used as a side inputs has a `KV` type.

      TODO(BEAM-115): Once Python SDK is compatible with the new Runner API,
      we could directly replace the coder instead of mutating the element type.
      """

            def visit_transform(self, transform_node):
                if False:
                    while True:
                        i = 10
                if isinstance(transform_node.transform, ParDo):
                    new_side_inputs = []
                    for side_input in transform_node.side_inputs:
                        access_pattern = side_input._side_input_data().access_pattern
                        if access_pattern == common_urns.side_inputs.ITERABLE.urn:
                            side_input.pvalue.element_type = typehints.Any
                            new_side_input = _DataflowIterableSideInput(side_input)
                        elif access_pattern == common_urns.side_inputs.MULTIMAP.urn:
                            side_input.pvalue.element_type = typehints.coerce_to_kv_type(side_input.pvalue.element_type, transform_node.full_label)
                            side_input.pvalue.requires_deterministic_key_coder = deterministic_key_coders and transform_node.full_label
                            new_side_input = _DataflowMultimapSideInput(side_input)
                        else:
                            raise ValueError('Unsupported access pattern for %r: %r' % (transform_node.full_label, access_pattern))
                        new_side_inputs.append(new_side_input)
                    transform_node.side_inputs = new_side_inputs
                    transform_node.transform.side_inputs = new_side_inputs
        return SideInputVisitor()

    @staticmethod
    def flatten_input_visitor():
        if False:
            while True:
                i = 10
        from apache_beam.pipeline import PipelineVisitor

        class FlattenInputVisitor(PipelineVisitor):
            """A visitor that replaces the element type for input ``PCollections``s of
       a ``Flatten`` transform with that of the output ``PCollection``.
      """

            def visit_transform(self, transform_node):
                if False:
                    while True:
                        i = 10
                from apache_beam import Flatten
                if isinstance(transform_node.transform, Flatten):
                    output_pcoll = DataflowRunner._only_element(transform_node.outputs.values())
                    for input_pcoll in transform_node.inputs:
                        input_pcoll.element_type = output_pcoll.element_type
        return FlattenInputVisitor()

    @staticmethod
    def combinefn_visitor():
        if False:
            while True:
                i = 10
        from apache_beam.pipeline import PipelineVisitor
        from apache_beam import core

        class CombineFnVisitor(PipelineVisitor):
            """Checks if `CombineFn` has non-default setup or teardown methods.
      If yes, raises `ValueError`.
      """

            def visit_transform(self, applied_transform):
                if False:
                    while True:
                        i = 10
                transform = applied_transform.transform
                if isinstance(transform, core.ParDo) and isinstance(transform.fn, core.CombineValuesDoFn):
                    if self._overrides_setup_or_teardown(transform.fn.combinefn):
                        raise ValueError('CombineFn.setup and CombineFn.teardown are not supported with non-portable Dataflow runner. Please use Dataflow Runner V2 instead.')

            @staticmethod
            def _overrides_setup_or_teardown(combinefn):
                if False:
                    for i in range(10):
                        print('nop')
                return False
        return CombineFnVisitor()

    def _adjust_pipeline_for_dataflow_v2(self, pipeline):
        if False:
            for i in range(10):
                print('nop')
        pipeline.visit(group_by_key_input_visitor(not pipeline._options.view_as(TypeOptions).allow_non_deterministic_key_coders))

    def run_pipeline(self, pipeline, options, pipeline_proto=None):
        if False:
            i = 10
            return i + 15
        'Remotely executes entire pipeline or parts reachable from node.'
        if _is_runner_v2_disabled(options):
            raise ValueError('Disabling Runner V2 no longer supported using Beam Python %s.' % beam.version.__version__)
        if is_in_notebook():
            notebook_version = 'goog-dataflow-notebook=' + beam.version.__version__.replace('.', '_')
            if options.view_as(GoogleCloudOptions).labels:
                options.view_as(GoogleCloudOptions).labels.append(notebook_version)
            else:
                options.view_as(GoogleCloudOptions).labels = [notebook_version]
        try:
            from apache_beam.runners.dataflow.internal import apiclient
        except ImportError:
            raise ImportError('Google Cloud Dataflow runner not available, please install apache_beam[gcp]')
        _check_and_add_missing_options(options)
        if pipeline:
            pipeline.visit(self.combinefn_visitor())
            pipeline.visit(self.side_input_visitor(deterministic_key_coders=not options.view_as(TypeOptions).allow_non_deterministic_key_coders))
            pipeline.replace_all(DataflowRunner._PTRANSFORM_OVERRIDES)
            if options.view_as(DebugOptions).lookup_experiment('use_legacy_bq_sink'):
                warnings.warn('Native sinks no longer implemented; ignoring use_legacy_bq_sink.')
        if pipeline_proto:
            self.proto_pipeline = pipeline_proto
        else:
            from apache_beam.transforms import environments
            if options.view_as(SetupOptions).prebuild_sdk_container_engine:
                self._default_environment = environments.DockerEnvironment.from_options(options)
                options.view_as(WorkerOptions).sdk_container_image = self._default_environment.container_image
            else:
                artifacts = environments.python_sdk_dependencies(options)
                if artifacts:
                    _LOGGER.info('Pipeline has additional dependencies to be installed in SDK worker container, consider using the SDK container image pre-building workflow to avoid repetitive installations. Learn more on https://cloud.google.com/dataflow/docs/guides/using-custom-containers#prebuild')
                self._default_environment = environments.DockerEnvironment.from_container_image(apiclient.get_container_image_from_options(options), artifacts=artifacts, resource_hints=environments.resource_hints_from_options(options))
            self._adjust_pipeline_for_dataflow_v2(pipeline)
            (self.proto_pipeline, self.proto_context) = pipeline.to_runner_api(return_context=True, default_environment=self._default_environment)
        if not options.view_as(StandardOptions).streaming:
            pre_optimize = options.view_as(DebugOptions).lookup_experiment('pre_optimize', 'default').lower()
            from apache_beam.runners.portability.fn_api_runner import translations
            if pre_optimize == 'none':
                phases = []
            elif pre_optimize == 'default' or pre_optimize == 'all':
                phases = [translations.pack_combiners, translations.sort_stages]
            else:
                phases = []
                for phase_name in pre_optimize.split(','):
                    if phase_name in ('pack_combiners',):
                        phases.append(getattr(translations, phase_name))
                    else:
                        raise ValueError('Unknown or inapplicable phase for pre_optimize: %s' % phase_name)
                phases.append(translations.sort_stages)
            if phases:
                self.proto_pipeline = translations.optimize_pipeline(self.proto_pipeline, phases=phases, known_runner_urns=frozenset(), partial=True)
        setup_options = options.view_as(SetupOptions)
        plugins = BeamPlugin.get_all_plugin_paths()
        if setup_options.beam_plugins is not None:
            plugins = list(set(plugins + setup_options.beam_plugins))
        setup_options.beam_plugins = plugins
        debug_options = options.view_as(DebugOptions)
        worker_options = options.view_as(WorkerOptions)
        if worker_options.min_cpu_platform:
            debug_options.add_experiment('min_cpu_platform=' + worker_options.min_cpu_platform)
        self.job = apiclient.Job(options, self.proto_pipeline)
        test_options = options.view_as(TestOptions)
        if test_options.dry_run:
            result = PipelineResult(PipelineState.DONE)
            result.wait_until_finish = lambda duration=None: None
            return result
        self.dataflow_client = apiclient.DataflowApplicationClient(options, self.job.root_staging_location)
        result = DataflowPipelineResult(self.dataflow_client.create_job(self.job), self)
        from apache_beam.runners.dataflow.dataflow_metrics import DataflowMetrics
        self._metrics = DataflowMetrics(self.dataflow_client, result, self.job)
        result.metric_results = self._metrics
        return result

    @staticmethod
    def _get_coder(typehint, window_coder):
        if False:
            for i in range(10):
                print('nop')
        'Returns a coder based on a typehint object.'
        if window_coder:
            return coders.WindowedValueCoder(coders.registry.get_coder(typehint), window_coder=window_coder)
        return coders.registry.get_coder(typehint)

    def _verify_gbk_coders(self, transform, pcoll):
        if False:
            print('Hello World!')
        parent = pcoll.producer
        if parent:
            coder = parent.transform._infer_output_coder()
        if not coder:
            coder = self._get_coder(pcoll.element_type or typehints.Any, None)
        if not coder.is_kv_coder():
            raise ValueError('Coder for the GroupByKey operation "%s" is not a key-value coder: %s.' % (transform.label, coder))
        coders.registry.verify_deterministic(coder.key_coder(), 'GroupByKey operation "%s"' % transform.label)

    def get_default_gcp_region(self):
        if False:
            while True:
                i = 10
        'Get a default value for Google Cloud region according to\n    https://cloud.google.com/compute/docs/gcloud-compute/#default-properties.\n    If no default can be found, returns None.\n    '
        environment_region = os.environ.get('CLOUDSDK_COMPUTE_REGION')
        if environment_region:
            _LOGGER.info('Using default GCP region %s from $CLOUDSDK_COMPUTE_REGION', environment_region)
            return environment_region
        try:
            cmd = ['gcloud', 'config', 'get-value', 'compute/region']
            raw_output = processes.check_output(cmd, stderr=DEVNULL)
            formatted_output = raw_output.decode('utf-8').strip()
            if formatted_output:
                _LOGGER.info('Using default GCP region %s from `%s`', formatted_output, ' '.join(cmd))
                return formatted_output
        except RuntimeError:
            pass
        return None

class _DataflowSideInput(beam.pvalue.AsSideInput):
    """Wraps a side input as a dataflow-compatible side input."""

    def _view_options(self):
        if False:
            return 10
        return {'data': self._data}

    def _side_input_data(self):
        if False:
            while True:
                i = 10
        return self._data

def _add_runner_v2_missing_options(options):
    if False:
        for i in range(10):
            print('nop')
    debug_options = options.view_as(DebugOptions)
    debug_options.add_experiment('beam_fn_api')
    debug_options.add_experiment('use_unified_worker')
    debug_options.add_experiment('use_runner_v2')
    debug_options.add_experiment('use_portable_job_submission')

def _check_and_add_missing_options(options):
    if False:
        for i in range(10):
            print('nop')
    'Validates and adds missing pipeline options depending on options set.\n\n  :param options: PipelineOptions for this pipeline.\n  '
    debug_options = options.view_as(DebugOptions)
    dataflow_service_options = options.view_as(GoogleCloudOptions).dataflow_service_options or []
    options.view_as(GoogleCloudOptions).dataflow_service_options = dataflow_service_options
    _add_runner_v2_missing_options(options)
    if 'enable_prime' in dataflow_service_options:
        debug_options.add_experiment('enable_prime')
    elif debug_options.lookup_experiment('enable_prime'):
        dataflow_service_options.append('enable_prime')
    sdk_location = options.view_as(SetupOptions).sdk_location
    if 'dev' in beam.version.__version__ and sdk_location == 'default':
        raise ValueError(f'You are submitting a pipeline with Apache Beam Python SDK {beam.version.__version__}. When launching Dataflow jobs with an unreleased (dev) SDK, please provide an SDK distribution in the --sdk_location option to use a consistent SDK version at pipeline submission and runtime. To ignore this error and use an SDK preinstalled in the default Dataflow dev runtime environment or in a custom container image, use --sdk_location=container.')
    if options.view_as(StandardOptions).streaming:
        google_cloud_options = options.view_as(GoogleCloudOptions)
        if not google_cloud_options.enable_streaming_engine and (debug_options.lookup_experiment('enable_windmill_service') or debug_options.lookup_experiment('enable_streaming_engine')):
            raise ValueError('Streaming engine both disabled and enabled:\n          --enable_streaming_engine flag is not set, but\n          enable_windmill_service and/or enable_streaming_engine experiments\n          are present. It is recommended you only set the\n          --enable_streaming_engine flag.')
        options.view_as(StandardOptions).streaming = True
        google_cloud_options.enable_streaming_engine = True
        debug_options.add_experiment('enable_streaming_engine')
        debug_options.add_experiment('enable_windmill_service')

def _is_runner_v2_disabled(options):
    if False:
        for i in range(10):
            print('nop')
    'Returns true if runner v2 is disabled.'
    debug_options = options.view_as(DebugOptions)
    return debug_options.lookup_experiment('disable_runner_v2') or debug_options.lookup_experiment('disable_runner_v2_until_2023') or debug_options.lookup_experiment('disable_runner_v2_until_v2.50') or debug_options.lookup_experiment('disable_prime_runner_v2')

class _DataflowIterableSideInput(_DataflowSideInput):
    """Wraps an iterable side input as dataflow-compatible side input."""

    def __init__(self, side_input):
        if False:
            i = 10
            return i + 15
        self.pvalue = side_input.pvalue
        side_input_data = side_input._side_input_data()
        assert side_input_data.access_pattern == common_urns.side_inputs.ITERABLE.urn
        self._data = beam.pvalue.SideInputData(common_urns.side_inputs.ITERABLE.urn, side_input_data.window_mapping_fn, side_input_data.view_fn)

class _DataflowMultimapSideInput(_DataflowSideInput):
    """Wraps a multimap side input as dataflow-compatible side input."""

    def __init__(self, side_input):
        if False:
            i = 10
            return i + 15
        self.pvalue = side_input.pvalue
        side_input_data = side_input._side_input_data()
        assert side_input_data.access_pattern == common_urns.side_inputs.MULTIMAP.urn
        self._data = beam.pvalue.SideInputData(common_urns.side_inputs.MULTIMAP.urn, side_input_data.window_mapping_fn, side_input_data.view_fn)

class DataflowPipelineResult(PipelineResult):
    """Represents the state of a pipeline run on the Dataflow service."""

    def __init__(self, job, runner):
        if False:
            print('Hello World!')
        'Initialize a new DataflowPipelineResult instance.\n\n    Args:\n      job: Job message from the Dataflow API. Could be :data:`None` if a job\n        request was not sent to Dataflow service (e.g. template jobs).\n      runner: DataflowRunner instance.\n    '
        self._job = job
        self._runner = runner
        self.metric_results = None

    def _update_job(self):
        if False:
            while True:
                i = 10
        if self.has_job and (not self.is_in_terminal_state()):
            self._job = self._runner.dataflow_client.get_job(self.job_id())

    def job_id(self):
        if False:
            while True:
                i = 10
        return self._job.id

    def metrics(self):
        if False:
            print('Hello World!')
        return self.metric_results

    def monitoring_infos(self):
        if False:
            return 10
        logging.warning('Monitoring infos not yet supported for Dataflow runner.')
        return []

    @property
    def has_job(self):
        if False:
            while True:
                i = 10
        return self._job is not None

    @staticmethod
    def api_jobstate_to_pipeline_state(api_jobstate):
        if False:
            print('Hello World!')
        values_enum = dataflow_api.Job.CurrentStateValueValuesEnum
        api_jobstate_map = defaultdict(lambda : PipelineState.UNRECOGNIZED, {values_enum.JOB_STATE_UNKNOWN: PipelineState.UNKNOWN, values_enum.JOB_STATE_STOPPED: PipelineState.STOPPED, values_enum.JOB_STATE_RUNNING: PipelineState.RUNNING, values_enum.JOB_STATE_DONE: PipelineState.DONE, values_enum.JOB_STATE_FAILED: PipelineState.FAILED, values_enum.JOB_STATE_CANCELLED: PipelineState.CANCELLED, values_enum.JOB_STATE_UPDATED: PipelineState.UPDATED, values_enum.JOB_STATE_DRAINING: PipelineState.DRAINING, values_enum.JOB_STATE_DRAINED: PipelineState.DRAINED, values_enum.JOB_STATE_PENDING: PipelineState.PENDING, values_enum.JOB_STATE_CANCELLING: PipelineState.CANCELLING, values_enum.JOB_STATE_RESOURCE_CLEANING_UP: PipelineState.RESOURCE_CLEANING_UP})
        return api_jobstate_map[api_jobstate] if api_jobstate else PipelineState.UNKNOWN

    def _get_job_state(self):
        if False:
            while True:
                i = 10
        return self.api_jobstate_to_pipeline_state(self._job.currentState)

    @property
    def state(self):
        if False:
            print('Hello World!')
        'Return the current state of the remote job.\n\n    Returns:\n      A PipelineState object.\n    '
        if not self.has_job:
            return PipelineState.UNKNOWN
        self._update_job()
        return self._get_job_state()

    def is_in_terminal_state(self):
        if False:
            print('Hello World!')
        if not self.has_job:
            return True
        return PipelineState.is_terminal(self._get_job_state())

    def wait_until_finish(self, duration=None):
        if False:
            i = 10
            return i + 15
        if not self.is_in_terminal_state():
            if not self.has_job:
                raise IOError('Failed to get the Dataflow job id.')
            consoleUrl = f'Console URL: https://console.cloud.google.com/dataflow/jobs/<RegionId>/{self.job_id()}?project=<ProjectId>'
            thread = threading.Thread(target=DataflowRunner.poll_for_job_completion, args=(self._runner, self, duration))
            thread.daemon = True
            thread.start()
            while thread.is_alive():
                time.sleep(5.0)
            terminated = self.is_in_terminal_state()
            assert duration or terminated, 'Job did not reach to a terminal state after waiting indefinitely. {}'.format(consoleUrl)
            if terminated and self.state != PipelineState.DONE:
                _LOGGER.error(consoleUrl)
                raise DataflowRuntimeException('Dataflow pipeline failed. State: %s, Error:\n%s' % (self.state, getattr(self._runner, 'last_error_msg', None)), self)
        elif PipelineState.is_terminal(self.state) and self.state == PipelineState.FAILED and self._runner:
            raise DataflowRuntimeException('Dataflow pipeline failed. State: %s, Error:\n%s' % (self.state, getattr(self._runner, 'last_error_msg', None)), self)
        return self.state

    def cancel(self):
        if False:
            return 10
        if not self.has_job:
            raise IOError('Failed to get the Dataflow job id.')
        self._update_job()
        if self.is_in_terminal_state():
            _LOGGER.warning('Cancel failed because job %s is already terminated in state %s.', self.job_id(), self.state)
        elif not self._runner.dataflow_client.modify_job_state(self.job_id(), 'JOB_STATE_CANCELLED'):
            cancel_failed_message = 'Failed to cancel job %s, please go to the Developers Console to cancel it manually.' % self.job_id()
            _LOGGER.error(cancel_failed_message)
            raise DataflowRuntimeException(cancel_failed_message, self)
        return self.state

    def __str__(self):
        if False:
            return 10
        return '<%s %s %s>' % (self.__class__.__name__, self.job_id(), self.state)

    def __repr__(self):
        if False:
            print('Hello World!')
        return '<%s %s at %s>' % (self.__class__.__name__, self._job, hex(id(self)))

class DataflowRuntimeException(Exception):
    """Indicates an error has occurred in running this pipeline."""

    def __init__(self, msg, result):
        if False:
            print('Hello World!')
        super().__init__(msg)
        self.result = result