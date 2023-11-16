"""A runner that allows running of Beam pipelines interactively.

This module is experimental. No backwards-compatibility guarantees.
"""
import logging
from typing import Optional
import apache_beam as beam
from apache_beam import runners
from apache_beam.options.pipeline_options import FlinkRunnerOptions
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import WorkerOptions
from apache_beam.pipeline import PipelineVisitor
from apache_beam.runners.direct import direct_runner
from apache_beam.runners.interactive import interactive_environment as ie
from apache_beam.runners.interactive import pipeline_instrument as inst
from apache_beam.runners.interactive import background_caching_job
from apache_beam.runners.interactive.dataproc.types import ClusterMetadata
from apache_beam.runners.interactive.display import pipeline_graph
from apache_beam.runners.interactive.options import capture_control
from apache_beam.runners.interactive.utils import to_element_list
from apache_beam.runners.interactive.utils import watch_sources
from apache_beam.testing.test_stream_service import TestStreamServiceController
SAMPLE_SIZE = 8
_LOGGER = logging.getLogger(__name__)

class InteractiveRunner(runners.PipelineRunner):
    """An interactive runner for Beam Python pipelines.

  Allows interactively building and running Beam Python pipelines.
  """

    def __init__(self, underlying_runner=None, render_option=None, skip_display=True, force_compute=True, blocking=True):
        if False:
            i = 10
            return i + 15
        'Constructor of InteractiveRunner.\n\n    Args:\n      underlying_runner: (runner.PipelineRunner)\n      render_option: (str) this parameter decides how the pipeline graph is\n          rendered. See display.pipeline_graph_renderer for available options.\n      skip_display: (bool) whether to skip display operations when running the\n          pipeline. Useful if running large pipelines when display is not\n          needed.\n      force_compute: (bool) whether sequential pipeline runs can use cached data\n          of PCollections computed from the previous runs including show API\n          invocation from interactive_beam module. If True, always run the whole\n          pipeline and compute data for PCollections forcefully. If False, use\n          available data and run minimum pipeline fragment to only compute data\n          not available.\n      blocking: (bool) whether the pipeline run should be blocking or not.\n    '
        self._underlying_runner = underlying_runner or direct_runner.DirectRunner()
        self._render_option = render_option
        self._in_session = False
        self._skip_display = skip_display
        self._force_compute = force_compute
        self._blocking = blocking

    def is_fnapi_compatible(self):
        if False:
            i = 10
            return i + 15
        return False

    def set_render_option(self, render_option):
        if False:
            return 10
        'Sets the rendering option.\n\n    Args:\n      render_option: (str) this parameter decides how the pipeline graph is\n          rendered. See display.pipeline_graph_renderer for available options.\n    '
        self._render_option = render_option

    def start_session(self):
        if False:
            while True:
                i = 10
        'Start the session that keeps back-end managers and workers alive.\n    '
        if self._in_session:
            return
        enter = getattr(self._underlying_runner, '__enter__', None)
        if enter is not None:
            _LOGGER.info('Starting session.')
            self._in_session = True
            enter()
        else:
            _LOGGER.error('Keep alive not supported.')

    def end_session(self):
        if False:
            for i in range(10):
                print('nop')
        'End the session that keeps backend managers and workers alive.\n    '
        if not self._in_session:
            return
        exit = getattr(self._underlying_runner, '__exit__', None)
        if exit is not None:
            self._in_session = False
            _LOGGER.info('Ending session.')
            exit(None, None, None)

    def apply(self, transform, pvalueish, options):
        if False:
            print('Hello World!')
        return self._underlying_runner.apply(transform, pvalueish, options)

    def run_pipeline(self, pipeline, options):
        if False:
            print('Hello World!')
        if not ie.current_env().options.enable_recording_replay:
            capture_control.evict_captured_data()
        if self._force_compute:
            ie.current_env().evict_computed_pcollections()
        watch_sources(pipeline)
        user_pipeline = ie.current_env().user_pipeline(pipeline)
        from apache_beam.runners.portability.flink_runner import FlinkRunner
        if isinstance(self._underlying_runner, FlinkRunner):
            self.configure_for_flink(user_pipeline, options)
        pipeline_instrument = inst.build_pipeline_instrument(pipeline, options)
        if user_pipeline:
            background_caching_job.attempt_to_run_background_caching_job(self._underlying_runner, user_pipeline, options)
            if background_caching_job.has_source_to_cache(user_pipeline) and (not background_caching_job.is_a_test_stream_service_running(user_pipeline)):
                streaming_cache_manager = ie.current_env().get_cache_manager(user_pipeline)
                if streaming_cache_manager and (not ie.current_env().get_test_stream_service_controller(user_pipeline)):

                    def exception_handler(e):
                        if False:
                            while True:
                                i = 10
                        _LOGGER.error(str(e))
                        return True
                    test_stream_service = TestStreamServiceController(streaming_cache_manager, exception_handler=exception_handler)
                    test_stream_service.start()
                    ie.current_env().set_test_stream_service_controller(user_pipeline, test_stream_service)
        pipeline_to_execute = beam.pipeline.Pipeline.from_runner_api(pipeline_instrument.instrumented_pipeline_proto(), self._underlying_runner, options)
        if ie.current_env().get_test_stream_service_controller(user_pipeline):
            endpoint = ie.current_env().get_test_stream_service_controller(user_pipeline).endpoint

            class TestStreamVisitor(PipelineVisitor):

                def visit_transform(self, transform_node):
                    if False:
                        return 10
                    from apache_beam.testing.test_stream import TestStream
                    if isinstance(transform_node.transform, TestStream) and (not transform_node.transform._events):
                        transform_node.transform._endpoint = endpoint
            pipeline_to_execute.visit(TestStreamVisitor())
        if not self._skip_display:
            a_pipeline_graph = pipeline_graph.PipelineGraph(pipeline_instrument.original_pipeline_proto, render_option=self._render_option)
            a_pipeline_graph.display_graph()
        main_job_result = PipelineResult(pipeline_to_execute.run(), pipeline_instrument)
        if user_pipeline:
            ie.current_env().set_pipeline_result(user_pipeline, main_job_result)
        if self._blocking:
            main_job_result.wait_until_finish()
        if main_job_result.state is beam.runners.runner.PipelineState.DONE:
            ie.current_env().mark_pcollection_computed(pipeline_instrument.cached_pcolls)
        return main_job_result

    def configure_for_flink(self, user_pipeline: beam.Pipeline, options: PipelineOptions) -> None:
        if False:
            while True:
                i = 10
        'Configures the pipeline options for running a job with Flink.\n\n    When running with a FlinkRunner, a job server started from an uber jar\n    (locally built or remotely downloaded) hosting the beam_job_api will\n    communicate with the Flink cluster located at the given flink_master in the\n    pipeline options.\n    '
        clusters = ie.current_env().clusters
        if clusters.pipelines.get(user_pipeline, None):
            return
        flink_master = self._strip_protocol_if_any(options.view_as(FlinkRunnerOptions).flink_master)
        cluster_metadata = clusters.default_cluster_metadata
        if flink_master == '[auto]':
            project_id = options.view_as(GoogleCloudOptions).project
            region = options.view_as(GoogleCloudOptions).region or 'us-central1'
            if project_id:
                if clusters.default_cluster_metadata:
                    cluster_metadata = ClusterMetadata(project_id=project_id, region=region, cluster_name=clusters.default_cluster_metadata.cluster_name)
                else:
                    cluster_metadata = ClusterMetadata(project_id=project_id, region=region)
                self._worker_options_to_cluster_metadata(options, cluster_metadata)
        elif flink_master in clusters.master_urls:
            cluster_metadata = clusters.cluster_metadata(flink_master)
        else:
            return
        if not cluster_metadata:
            return
        dcm = clusters.create(cluster_metadata)
        clusters.pipelines[user_pipeline] = dcm
        dcm.pipelines.add(user_pipeline)
        self._configure_flink_options(options, clusters.DATAPROC_FLINK_VERSION, dcm.cluster_metadata.master_url)

    def _strip_protocol_if_any(self, flink_master: Optional[str]):
        if False:
            i = 10
            return i + 15
        if flink_master:
            parts = flink_master.split('://')
            if len(parts) > 1:
                return parts[1]
        return flink_master

    def _worker_options_to_cluster_metadata(self, options: PipelineOptions, cluster_metadata: ClusterMetadata):
        if False:
            return 10
        worker_options = options.view_as(WorkerOptions)
        if worker_options.subnetwork:
            cluster_metadata.subnetwork = worker_options.subnetwork
        if worker_options.num_workers:
            cluster_metadata.num_workers = worker_options.num_workers
        if worker_options.machine_type:
            cluster_metadata.machine_type = worker_options.machine_type

    def _configure_flink_options(self, options: PipelineOptions, flink_version: str, master_url: str):
        if False:
            while True:
                i = 10
        flink_options = options.view_as(FlinkRunnerOptions)
        flink_options.flink_version = flink_version
        flink_options.flink_master = master_url

class PipelineResult(beam.runners.runner.PipelineResult):
    """Provides access to information about a pipeline."""

    def __init__(self, underlying_result, pipeline_instrument):
        if False:
            while True:
                i = 10
        'Constructor of PipelineResult.\n\n    Args:\n      underlying_result: (PipelineResult) the result returned by the underlying\n          runner running the pipeline.\n      pipeline_instrument: (PipelineInstrument) pipeline instrument describing\n          the pipeline being executed with interactivity applied and related\n          metadata including where the interactivity-backing cache lies.\n    '
        super().__init__(underlying_result.state)
        self._underlying_result = underlying_result
        self._pipeline_instrument = pipeline_instrument

    @property
    def state(self):
        if False:
            return 10
        return self._underlying_result.state

    def wait_until_finish(self):
        if False:
            return 10
        self._underlying_result.wait_until_finish()

    def get(self, pcoll, include_window_info=False):
        if False:
            i = 10
            return i + 15
        'Materializes the PCollection into a list.\n\n    If include_window_info is True, then returns the elements as\n    WindowedValues. Otherwise, return the element as itself.\n    '
        return list(self.read(pcoll, include_window_info))

    def read(self, pcoll, include_window_info=False):
        if False:
            while True:
                i = 10
        'Reads the PCollection one element at a time from cache.\n\n    If include_window_info is True, then returns the elements as\n    WindowedValues. Otherwise, return the element as itself.\n    '
        key = self._pipeline_instrument.cache_key(pcoll)
        cache_manager = ie.current_env().get_cache_manager(self._pipeline_instrument.user_pipeline)
        if key and cache_manager.exists('full', key):
            coder = cache_manager.load_pcoder('full', key)
            (reader, _) = cache_manager.read('full', key)
            return to_element_list(reader, coder, include_window_info)
        else:
            raise ValueError('PCollection not available, please run the pipeline.')

    def cancel(self):
        if False:
            print('Hello World!')
        self._underlying_result.cancel()