"""SDK Fn Harness entry point."""
import importlib
import json
import logging
import os
import re
import sys
import traceback
from google.protobuf import text_format
from apache_beam.internal import pickler
from apache_beam.io import filesystems
from apache_beam.options.pipeline_options import DebugOptions
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import ProfilingOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.options.pipeline_options import WorkerOptions
from apache_beam.options.value_provider import RuntimeValueProvider
from apache_beam.portability.api import endpoints_pb2
from apache_beam.runners.internal import names
from apache_beam.runners.worker.data_sampler import DataSampler
from apache_beam.runners.worker.log_handler import FnApiLogRecordHandler
from apache_beam.runners.worker.sdk_worker import SdkHarness
from apache_beam.utils import profiler
_LOGGER = logging.getLogger(__name__)
_ENABLE_GOOGLE_CLOUD_PROFILER = 'enable_google_cloud_profiler'

def _import_beam_plugins(plugins):
    if False:
        while True:
            i = 10
    for plugin in plugins:
        try:
            importlib.import_module(plugin)
            _LOGGER.debug('Imported beam-plugin %s', plugin)
        except ImportError:
            try:
                _LOGGER.debug("Looks like %s is not a module. Trying to import it assuming it's a class", plugin)
                (module, _) = plugin.rsplit('.', 1)
                importlib.import_module(module)
                _LOGGER.debug('Imported %s for beam-plugin %s', module, plugin)
            except ImportError as exc:
                _LOGGER.warning('Failed to import beam-plugin %s', plugin, exc_info=exc)

def create_harness(environment, dry_run=False):
    if False:
        for i in range(10):
            print('nop')
    'Creates SDK Fn Harness.'
    deferred_exception = None
    if 'LOGGING_API_SERVICE_DESCRIPTOR' in environment:
        try:
            logging_service_descriptor = endpoints_pb2.ApiServiceDescriptor()
            text_format.Merge(environment['LOGGING_API_SERVICE_DESCRIPTOR'], logging_service_descriptor)
            fn_log_handler = FnApiLogRecordHandler(logging_service_descriptor)
            logging.getLogger().addHandler(fn_log_handler)
            _LOGGER.info('Logging handler created.')
        except Exception:
            _LOGGER.error('Failed to set up logging handler, continuing without.', exc_info=True)
            fn_log_handler = None
    else:
        fn_log_handler = None
    pipeline_options_dict = _load_pipeline_options(environment.get('PIPELINE_OPTIONS'))
    default_log_level = _get_log_level_from_options_dict(pipeline_options_dict)
    logging.getLogger().setLevel(default_log_level)
    _set_log_level_overrides(pipeline_options_dict)
    RuntimeValueProvider.set_runtime_options(pipeline_options_dict)
    sdk_pipeline_options = PipelineOptions.from_dictionary(pipeline_options_dict)
    filesystems.FileSystems.set_options(sdk_pipeline_options)
    pickle_library = sdk_pipeline_options.view_as(SetupOptions).pickle_library
    pickler.set_library(pickle_library)
    if 'SEMI_PERSISTENT_DIRECTORY' in environment:
        semi_persistent_directory = environment['SEMI_PERSISTENT_DIRECTORY']
    else:
        semi_persistent_directory = None
    _LOGGER.info('semi_persistent_directory: %s', semi_persistent_directory)
    _worker_id = environment.get('WORKER_ID', None)
    if pickle_library != pickler.USE_CLOUDPICKLE:
        try:
            _load_main_session(semi_persistent_directory)
        except LoadMainSessionException:
            exception_details = traceback.format_exc()
            _LOGGER.error('Could not load main session: %s', exception_details, exc_info=True)
            raise
        except Exception:
            summary = 'Could not load main session. Inspect which external dependencies are used in the main module of your pipeline. Verify that corresponding packages are installed in the pipeline runtime environment and their installed versions match the versions used in pipeline submission environment. For more information, see: https://beam.apache.org/documentation/sdks/python-pipeline-dependencies/'
            _LOGGER.error(summary, exc_info=True)
            exception_details = traceback.format_exc()
            deferred_exception = LoadMainSessionException(f'{summary} {exception_details}')
    _LOGGER.info('Pipeline_options: %s', sdk_pipeline_options.get_all_options(drop_default=True))
    control_service_descriptor = endpoints_pb2.ApiServiceDescriptor()
    status_service_descriptor = endpoints_pb2.ApiServiceDescriptor()
    text_format.Merge(environment['CONTROL_API_SERVICE_DESCRIPTOR'], control_service_descriptor)
    if 'STATUS_API_SERVICE_DESCRIPTOR' in environment:
        text_format.Merge(environment['STATUS_API_SERVICE_DESCRIPTOR'], status_service_descriptor)
    assert not control_service_descriptor.HasField('authentication')
    experiments = sdk_pipeline_options.view_as(DebugOptions).experiments or []
    enable_heap_dump = 'enable_heap_dump' in experiments
    beam_plugins = sdk_pipeline_options.view_as(SetupOptions).beam_plugins or []
    _import_beam_plugins(beam_plugins)
    if dry_run:
        return
    data_sampler = DataSampler.create(sdk_pipeline_options)
    sdk_harness = SdkHarness(control_address=control_service_descriptor.url, status_address=status_service_descriptor.url, worker_id=_worker_id, state_cache_size=_get_state_cache_size_bytes(options=sdk_pipeline_options), data_buffer_time_limit_ms=_get_data_buffer_time_limit_ms(experiments), profiler_factory=profiler.Profile.factory_from_options(sdk_pipeline_options.view_as(ProfilingOptions)), enable_heap_dump=enable_heap_dump, data_sampler=data_sampler, deferred_exception=deferred_exception)
    return (fn_log_handler, sdk_harness, sdk_pipeline_options)

def _start_profiler(gcp_profiler_service_name, gcp_profiler_service_version):
    if False:
        print('Hello World!')
    try:
        import googlecloudprofiler
        if gcp_profiler_service_name and gcp_profiler_service_version:
            googlecloudprofiler.start(service=gcp_profiler_service_name, service_version=gcp_profiler_service_version, verbose=1)
            _LOGGER.info('Turning on Google Cloud Profiler.')
        else:
            raise RuntimeError('Unable to find the job id or job name from envvar.')
    except Exception as e:
        _LOGGER.warning('Unable to start google cloud profiler due to error: %s. For how to enable Cloud Profiler with Dataflow see https://cloud.google.com/dataflow/docs/guides/profiling-a-pipeline.For troubleshooting tips with Cloud Profiler see https://cloud.google.com/profiler/docs/troubleshooting.' % e)

def _get_gcp_profiler_name_if_enabled(sdk_pipeline_options):
    if False:
        while True:
            i = 10
    gcp_profiler_service_name = sdk_pipeline_options.view_as(GoogleCloudOptions).get_cloud_profiler_service_name()
    return gcp_profiler_service_name

def main(unused_argv):
    if False:
        i = 10
        return i + 15
    'Main entry point for SDK Fn Harness.'
    (fn_log_handler, sdk_harness, sdk_pipeline_options) = create_harness(os.environ)
    gcp_profiler_name = _get_gcp_profiler_name_if_enabled(sdk_pipeline_options)
    if gcp_profiler_name:
        _start_profiler(gcp_profiler_name, os.environ['JOB_ID'])
    try:
        _LOGGER.info('Python sdk harness starting.')
        sdk_harness.run()
        _LOGGER.info('Python sdk harness exiting.')
    except:
        _LOGGER.critical('Python sdk harness failed: ', exc_info=True)
        raise
    finally:
        if fn_log_handler:
            fn_log_handler.close()

def _load_pipeline_options(options_json):
    if False:
        for i in range(10):
            print('nop')
    if options_json is None:
        return {}
    options = json.loads(options_json)
    if 'options' in options:
        return options.get('options')
    else:
        portable_option_regex = '^beam:option:(?P<key>.*):v1$'
        return {re.match(portable_option_regex, k).group('key') if re.match(portable_option_regex, k) else k: v for (k, v) in options.items()}

def _parse_pipeline_options(options_json):
    if False:
        i = 10
        return i + 15
    return PipelineOptions.from_dictionary(_load_pipeline_options(options_json))

def _get_state_cache_size_bytes(options):
    if False:
        for i in range(10):
            print('nop')
    'Return the maximum size of state cache in bytes.\n\n  Returns:\n    an int indicating the maximum number of bytes to cache.\n  '
    max_cache_memory_usage_mb = options.view_as(WorkerOptions).max_cache_memory_usage_mb
    experiments = options.view_as(DebugOptions).experiments or []
    for experiment in experiments:
        if re.match('state_cache_size=', experiment):
            _LOGGER.warning('--experiments=state_cache_size=X is deprecated and will be removed in future releases.Please use --max_cache_memory_usage_mb=X to set the cache size for user state API and side inputs.')
            return int(re.match('state_cache_size=(?P<state_cache_size>.*)', experiment).group('state_cache_size')) << 20
    return max_cache_memory_usage_mb << 20

def _get_data_buffer_time_limit_ms(experiments):
    if False:
        return 10
    'Defines the time limt of the outbound data buffering.\n\n  Note: data_buffer_time_limit_ms is an experimental flag and might\n  not be available in future releases.\n\n  Returns:\n    an int indicating the time limit in milliseconds of the outbound\n      data buffering. Default is 0 (disabled)\n  '
    for experiment in experiments:
        if re.match('data_buffer_time_limit_ms=', experiment):
            return int(re.match('data_buffer_time_limit_ms=(?P<data_buffer_time_limit_ms>.*)', experiment).group('data_buffer_time_limit_ms'))
    return 0

def _get_log_level_from_options_dict(options_dict: dict) -> int:
    if False:
        print('Hello World!')
    "Get log level from options dict's entry `default_sdk_harness_log_level`.\n  If not specified, default log level is logging.INFO.\n  "
    dict_level = options_dict.get('default_sdk_harness_log_level', 'INFO')
    log_level = dict_level
    if log_level.isdecimal():
        log_level = int(log_level)
    else:
        log_level = getattr(logging, log_level, None)
        if not isinstance(log_level, int):
            _LOGGER.error('Unknown log level %s. Use default value INFO.', dict_level)
            log_level = logging.INFO
    return log_level

def _set_log_level_overrides(options_dict: dict) -> None:
    if False:
        print('Hello World!')
    "Set module log level overrides from options dict's entry\n  `sdk_harness_log_level_overrides`.\n  "
    parsed_overrides = options_dict.get('sdk_harness_log_level_overrides', None)
    if not isinstance(parsed_overrides, dict):
        if parsed_overrides is not None:
            _LOGGER.error('Unable to parse sdk_harness_log_level_overrides: %s', parsed_overrides)
        return
    for (module_name, log_level) in parsed_overrides.items():
        try:
            logging.getLogger(module_name).setLevel(log_level)
        except Exception as e:
            _LOGGER.error('Error occurred when setting log level for %s: %s', module_name, e)

class LoadMainSessionException(Exception):
    """
  Used to crash this worker if a main session file failed to load.
  """
    pass

def _load_main_session(semi_persistent_directory):
    if False:
        while True:
            i = 10
    'Loads a pickled main session from the path specified.'
    if semi_persistent_directory:
        session_file = os.path.join(semi_persistent_directory, 'staged', names.PICKLED_MAIN_SESSION_FILE)
        if os.path.isfile(session_file):
            if os.path.getsize(session_file) == 0:
                raise LoadMainSessionException('Session file found, but empty: %s. Functions defined in __main__ (interactive session) will almost certainly fail.' % (session_file,))
            pickler.load_session(session_file)
        else:
            _LOGGER.warning('No session file found: %s. Functions defined in __main__ (interactive session) may fail.', session_file)
    else:
        _LOGGER.warning('No semi_persistent_directory found: Functions defined in __main__ (interactive session) may fail.')
if __name__ == '__main__':
    main(sys.argv)