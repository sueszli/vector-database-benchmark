"""Module to build and run background source recording jobs.

For internal use only; no backwards-compatibility guarantees.

A background source recording job is a job that records events for all
recordable sources of a given pipeline. With Interactive Beam, one such job is
started when a pipeline run happens (which produces a main job in contrast to
the background source recording job) and meets the following conditions:

  #. The pipeline contains recordable sources, configured through
     interactive_beam.options.recordable_sources.
  #. No such background job is running.
  #. No such background job has completed successfully and the cached events are
     still valid (invalidated when recordable sources change in the pipeline).

Once started, the background source recording job runs asynchronously until it
hits some recording limit configured in interactive_beam.options. Meanwhile,
the main job and future main jobs from the pipeline will run using the
deterministic replayable recorded events until they are invalidated.
"""
import logging
import threading
import time
import apache_beam as beam
from apache_beam.runners.interactive import interactive_environment as ie
from apache_beam.runners.interactive import utils
from apache_beam.runners.interactive.caching import streaming_cache
from apache_beam.runners.runner import PipelineState
_LOGGER = logging.getLogger(__name__)

class BackgroundCachingJob(object):
    """A simple abstraction that controls necessary components of a timed and
  space limited background source recording job.

  A background source recording job successfully completes source data
  recording in 2 conditions:

    #. The job is finite and runs into DONE state;
    #. The job is infinite but hits an interactive_beam.options configured limit
       and gets cancelled into CANCELLED/CANCELLING state.

  In both situations, the background source recording job should be treated as
  done successfully.
  """

    def __init__(self, pipeline_result, limiters):
        if False:
            for i in range(10):
                print('nop')
        self._pipeline_result = pipeline_result
        self._result_lock = threading.RLock()
        self._condition_checker = threading.Thread(target=self._background_caching_job_condition_checker, daemon=True)
        self._limiters = limiters
        self._condition_checker.start()

    def _background_caching_job_condition_checker(self):
        if False:
            while True:
                i = 10
        while True:
            with self._result_lock:
                if PipelineState.is_terminal(self._pipeline_result.state):
                    break
            if self._should_end_condition_checker():
                self.cancel()
                break
            time.sleep(0.5)

    def _should_end_condition_checker(self):
        if False:
            return 10
        return any((l.is_triggered() for l in self._limiters))

    def is_done(self):
        if False:
            return 10
        with self._result_lock:
            is_terminated = self._pipeline_result.state in (PipelineState.DONE, PipelineState.CANCELLED)
            is_triggered = self._should_end_condition_checker()
            is_cancelling = self._pipeline_result.state is PipelineState.CANCELLING
        return is_terminated or (is_triggered and is_cancelling)

    def is_running(self):
        if False:
            while True:
                i = 10
        with self._result_lock:
            return self._pipeline_result.state is PipelineState.RUNNING

    def cancel(self):
        if False:
            for i in range(10):
                print('nop')
        'Cancels this background source recording job.\n    '
        with self._result_lock:
            if not PipelineState.is_terminal(self._pipeline_result.state):
                try:
                    self._pipeline_result.cancel()
                except NotImplementedError:
                    pass

    @property
    def state(self):
        if False:
            while True:
                i = 10
        with self._result_lock:
            return self._pipeline_result.state

def attempt_to_run_background_caching_job(runner, user_pipeline, options=None, limiters=None):
    if False:
        return 10
    'Attempts to run a background source recording job for a user-defined\n  pipeline.\n\n  Returns True if a job was started, False otherwise.\n\n  The pipeline result is automatically tracked by Interactive Beam in case\n  future cancellation/cleanup is needed.\n  '
    if is_background_caching_job_needed(user_pipeline):
        attempt_to_cancel_background_caching_job(user_pipeline)
        attempt_to_stop_test_stream_service(user_pipeline)
        from apache_beam.runners.interactive import pipeline_instrument as instr
        runner_pipeline = beam.pipeline.Pipeline.from_runner_api(user_pipeline.to_runner_api(), runner, options)
        ie.current_env().add_derived_pipeline(user_pipeline, runner_pipeline)
        background_caching_job_result = beam.pipeline.Pipeline.from_runner_api(instr.build_pipeline_instrument(runner_pipeline).background_caching_pipeline_proto(), runner, options).run()
        recording_limiters = limiters if limiters else ie.current_env().options.capture_control.limiters()
        ie.current_env().set_background_caching_job(user_pipeline, BackgroundCachingJob(background_caching_job_result, limiters=recording_limiters))
        return True
    return False

def is_background_caching_job_needed(user_pipeline):
    if False:
        return 10
    'Determines if a background source recording job needs to be started.\n\n  It does several state checks and recording state changes throughout the\n  process. It is not idempotent to simplify the usage.\n  '
    job = ie.current_env().get_background_caching_job(user_pipeline)
    need_cache = has_source_to_cache(user_pipeline)
    cache_changed = is_source_to_cache_changed(user_pipeline)
    if need_cache and (not ie.current_env().options.enable_recording_replay):
        from apache_beam.runners.interactive.options import capture_control
        capture_control.evict_captured_data()
        return True
    return need_cache and (not job or not (job.is_done() or job.is_running()) or cache_changed)

def is_cache_complete(pipeline_id):
    if False:
        while True:
            i = 10
    'Returns True if the backgrond cache for the given pipeline is done.\n  '
    user_pipeline = ie.current_env().pipeline_id_to_pipeline(pipeline_id)
    job = ie.current_env().get_background_caching_job(user_pipeline)
    is_done = job and job.is_done()
    cache_changed = is_source_to_cache_changed(user_pipeline, update_cached_source_signature=False)
    return is_done or cache_changed

def has_source_to_cache(user_pipeline):
    if False:
        while True:
            i = 10
    "Determines if a user-defined pipeline contains any source that need to be\n  cached. If so, also immediately wrap current cache manager held by current\n  interactive environment into a streaming cache if this has not been done.\n  The wrapping doesn't invalidate existing cache in any way.\n\n  This can help determining if a background source recording job is needed to\n  write cache for sources and if a test stream service is needed to serve the\n  cache.\n\n  Throughout the check, if source-to-cache has changed from the last check, it\n  also cleans up the invalidated cache early on.\n  "
    has_cache = utils.has_unbounded_sources(user_pipeline)
    if has_cache:
        if not isinstance(ie.current_env().get_cache_manager(user_pipeline, create_if_absent=True), streaming_cache.StreamingCache):
            file_based_cm = ie.current_env().get_cache_manager(user_pipeline)
            cache_dir = file_based_cm._cache_dir
            cache_root = ie.current_env().options.cache_root
            if cache_root:
                if cache_root.startswith('gs://'):
                    raise ValueError('GCS cache paths are not currently supported for streaming pipelines.')
                cache_dir = cache_root
            ie.current_env().set_cache_manager(streaming_cache.StreamingCache(cache_dir, is_cache_complete=is_cache_complete, sample_resolution_sec=1.0, saved_pcoders=file_based_cm._saved_pcoders), user_pipeline)
    return has_cache

def attempt_to_cancel_background_caching_job(user_pipeline):
    if False:
        print('Hello World!')
    'Attempts to cancel background source recording job for a user-defined\n  pipeline.\n\n  If no background source recording job needs to be cancelled, NOOP. Otherwise,\n  cancel such job.\n  '
    job = ie.current_env().get_background_caching_job(user_pipeline)
    if job:
        job.cancel()

def attempt_to_stop_test_stream_service(user_pipeline):
    if False:
        for i in range(10):
            print('nop')
    'Attempts to stop the gRPC server/service serving the test stream.\n\n  If there is no such server started, NOOP. Otherwise, stop it.\n  '
    if is_a_test_stream_service_running(user_pipeline):
        ie.current_env().evict_test_stream_service_controller(user_pipeline).stop()

def is_a_test_stream_service_running(user_pipeline):
    if False:
        for i in range(10):
            print('nop')
    'Checks to see if there is a gPRC server/service running that serves the\n  test stream to any job started from the given user_pipeline.\n  '
    return ie.current_env().get_test_stream_service_controller(user_pipeline) is not None

def is_source_to_cache_changed(user_pipeline, update_cached_source_signature=True):
    if False:
        return 10
    "Determines if there is any change in the sources that need to be cached\n  used by the user-defined pipeline.\n\n  Due to the expensiveness of computations and for the simplicity of usage, this\n  function is not idempotent because Interactive Beam automatically discards\n  previously tracked signature of transforms and tracks the current signature of\n  transforms for the user-defined pipeline if there is any change.\n\n  When it's True, there is addition/deletion/mutation of source transforms that\n  requires a new background source recording job.\n  "
    recorded_signature = ie.current_env().get_cached_source_signature(user_pipeline)
    current_signature = extract_source_to_cache_signature(user_pipeline)
    is_changed = not current_signature.issubset(recorded_signature)
    if is_changed and update_cached_source_signature:
        options = ie.current_env().options
        if options.enable_recording_replay:
            if not recorded_signature:

                def sizeof_fmt(num, suffix='B'):
                    if False:
                        print('Hello World!')
                    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
                        if abs(num) < 1000.0:
                            return '%3.1f%s%s' % (num, unit, suffix)
                        num /= 1000.0
                    return '%.1f%s%s' % (num, 'Yi', suffix)
                _LOGGER.info('Interactive Beam has detected unbounded sources in your pipeline. In order to have a deterministic replay, a segment of data will be recorded from all sources for %s seconds or until a total of %s have been written to disk.', options.recording_duration.total_seconds(), sizeof_fmt(options.recording_size_limit))
            else:
                _LOGGER.info('Interactive Beam has detected a new streaming source was added to the pipeline. In order for the cached streaming data to start at the same time, all recorded data has been cleared and a new segment of data will be recorded.')
        ie.current_env().cleanup(user_pipeline)
        ie.current_env().set_cached_source_signature(user_pipeline, current_signature)
        ie.current_env().add_user_pipeline(user_pipeline)
    return is_changed

def extract_source_to_cache_signature(user_pipeline):
    if False:
        return 10
    'Extracts a set of signature for sources that need to be cached in the\n  user-defined pipeline.\n\n  A signature is a str representation of urn and payload of a source.\n  '
    unbounded_sources_as_applied_transforms = utils.unbounded_sources(user_pipeline)
    unbounded_sources_as_ptransforms = set(map(lambda x: x.transform, unbounded_sources_as_applied_transforms))
    (_, context) = user_pipeline.to_runner_api(return_context=True)
    signature = set(map(lambda transform: str(transform.to_runner_api(context)), unbounded_sources_as_ptransforms))
    return signature