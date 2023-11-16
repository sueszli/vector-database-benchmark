"""Module of the current Interactive Beam environment.

For internal use only; no backwards-compatibility guarantees.
Provides interfaces to interact with existing Interactive Beam environment.
External Interactive Beam users please use interactive_beam module in
application code or notebook.
"""
import atexit
import importlib
import logging
import os
import tempfile
from collections.abc import Iterable
from pathlib import PurePath
import apache_beam as beam
from apache_beam.runners import DataflowRunner
from apache_beam.runners import runner
from apache_beam.runners.direct import direct_runner
from apache_beam.runners.interactive import cache_manager as cache
from apache_beam.runners.interactive.messaging.interactive_environment_inspector import InteractiveEnvironmentInspector
from apache_beam.runners.interactive.recording_manager import RecordingManager
from apache_beam.runners.interactive.sql.sql_chain import SqlChain
from apache_beam.runners.interactive.user_pipeline_tracker import UserPipelineTracker
from apache_beam.runners.interactive.utils import assert_bucket_exists
from apache_beam.runners.interactive.utils import detect_pipeline_runner
from apache_beam.runners.interactive.utils import register_ipython_log_handler
from apache_beam.utils.interactive_utils import is_in_ipython
from apache_beam.utils.interactive_utils import is_in_notebook
_interactive_beam_env = None
_LOGGER = logging.getLogger(__name__)
_JQUERY_WITH_DATATABLE_TEMPLATE = "\n        if (typeof window.interactive_beam_jquery == 'undefined') {{\n          var jqueryScript = document.createElement('script');\n          jqueryScript.src = 'https://code.jquery.com/jquery-3.4.1.slim.min.js';\n          jqueryScript.type = 'text/javascript';\n          jqueryScript.onload = function() {{\n            var datatableScript = document.createElement('script');\n            datatableScript.src = 'https://cdn.datatables.net/1.10.20/js/jquery.dataTables.min.js';\n            datatableScript.type = 'text/javascript';\n            datatableScript.onload = function() {{\n              window.interactive_beam_jquery = jQuery.noConflict(true);\n              window.interactive_beam_jquery(document).ready(function($){{\n                {customized_script}\n              }});\n            }}\n            document.head.appendChild(datatableScript);\n          }};\n          document.head.appendChild(jqueryScript);\n        }} else {{\n          window.interactive_beam_jquery(document).ready(function($){{\n            {customized_script}\n          }});\n        }}"
_HTML_IMPORT_TEMPLATE = "\n        var import_html = () => {{\n          {hrefs}.forEach(href => {{\n            var link = document.createElement('link');\n            link.rel = 'import'\n            link.href = href;\n            document.head.appendChild(link);\n          }});\n        }}\n        if ('import' in document.createElement('link')) {{\n          import_html();\n        }} else {{\n          var webcomponentScript = document.createElement('script');\n          webcomponentScript.src = 'https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/1.3.3/webcomponents-lite.js';\n          webcomponentScript.type = 'text/javascript';\n          webcomponentScript.onload = function(){{\n            import_html();\n          }};\n          document.head.appendChild(webcomponentScript);\n        }}"

def current_env():
    if False:
        print('Hello World!')
    'Gets current Interactive Beam environment.'
    global _interactive_beam_env
    if not _interactive_beam_env:
        _interactive_beam_env = InteractiveEnvironment()
    return _interactive_beam_env

def new_env():
    if False:
        return 10
    'Creates a new Interactive Beam environment to replace current one.'
    global _interactive_beam_env
    if _interactive_beam_env:
        _interactive_beam_env.cleanup()
    _interactive_beam_env = None
    return current_env()

class InteractiveEnvironment(object):
    """An interactive environment with cache and pipeline variable metadata.

  Interactive Beam will use the watched variable information to determine if a
  PCollection is assigned to a variable in user pipeline definition. When
  executing the pipeline, interactivity is applied with implicit cache
  mechanism for those PCollections if the pipeline is interactive. Users can
  also visualize and introspect those PCollections in user code since they have
  handles to the variables.
  """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        atexit.register(self.cleanup)
        self._cache_managers = {}
        self._recording_managers = {}
        self._watching_set = set()
        self._watching_dict_list = []
        self._main_pipeline_results = {}
        self._background_caching_jobs = {}
        self._test_stream_service_controllers = {}
        self._cached_source_signature = {}
        self._tracked_user_pipelines = UserPipelineTracker()
        from apache_beam.runners.interactive.interactive_beam import clusters
        self.clusters = clusters
        self._computed_pcolls = set()
        self.watch('__main__')
        try:
            import IPython
            import timeloop
            from facets_overview.generic_feature_statistics_generator import GenericFeatureStatisticsGenerator
            from google.cloud import dataproc_v1
            self._is_interactive_ready = True
        except ImportError:
            self._is_interactive_ready = False
            _LOGGER.warning('Dependencies required for Interactive Beam PCollection visualization are not available, please use: `pip install apache-beam[interactive]` to install necessary dependencies to enable all data visualization features.')
        self._is_in_ipython = is_in_ipython()
        self._is_in_notebook = is_in_notebook()
        if not self._is_in_ipython:
            _LOGGER.warning('You cannot use Interactive Beam features when you are not in an interactive environment such as a Jupyter notebook or ipython terminal.')
        if self._is_in_ipython and (not self._is_in_notebook):
            _LOGGER.warning('You have limited Interactive Beam features since your ipython kernel is not connected to any notebook frontend.')
        if self._is_in_notebook:
            self.load_jquery_with_datatable()
            register_ipython_log_handler()
        self._inspector = InteractiveEnvironmentInspector()
        self._inspector_with_synthetic = InteractiveEnvironmentInspector(ignore_synthetic=False)
        self.sql_chain = {}

    @property
    def options(self):
        if False:
            for i in range(10):
                print('nop')
        'A reference to the global interactive options.\n\n    Provided to avoid import loop or excessive dynamic import. All internal\n    Interactive Beam modules should access interactive_beam.options through\n    this property.\n    '
        from apache_beam.runners.interactive.interactive_beam import options
        return options

    @property
    def is_interactive_ready(self):
        if False:
            for i in range(10):
                print('nop')
        'If the [interactive] dependencies are installed.'
        return self._is_interactive_ready

    @property
    def is_in_ipython(self):
        if False:
            for i in range(10):
                print('nop')
        'If the runtime is within an IPython kernel.'
        return self._is_in_ipython

    @property
    def is_in_notebook(self):
        if False:
            return 10
        'If the kernel is connected to a notebook frontend.\n\n    If not, it could be that the user is using kernel in a terminal or a unit\n    test.\n    '
        return self._is_in_notebook

    @property
    def inspector(self):
        if False:
            print('Hello World!')
        'Gets the singleton InteractiveEnvironmentInspector to retrieve\n    information consumable by other applications such as a notebook\n    extension.'
        return self._inspector

    @property
    def inspector_with_synthetic(self):
        if False:
            while True:
                i = 10
        'Gets the singleton InteractiveEnvironmentInspector with additional\n    synthetic variables generated by Interactive Beam. Internally used.'
        return self._inspector_with_synthetic

    def cleanup_pipeline(self, pipeline):
        if False:
            while True:
                i = 10
        from apache_beam.runners.interactive import background_caching_job as bcj
        bcj.attempt_to_cancel_background_caching_job(pipeline)
        bcj.attempt_to_stop_test_stream_service(pipeline)
        cache_manager = self.get_cache_manager(pipeline)
        if cache_manager and self.get_recording_manager(pipeline) is None:
            cache_manager.cleanup()
        self.clusters.cleanup(pipeline)

    def cleanup_environment(self):
        if False:
            i = 10
            return i + 15
        for (_, job) in self._background_caching_jobs.items():
            if job:
                job.cancel()
        for (_, controller) in self._test_stream_service_controllers.items():
            if controller:
                controller.stop()
        for (pipeline_id, cache_manager) in self._cache_managers.items():
            if cache_manager and pipeline_id not in self._recording_managers:
                cache_manager.cleanup()
        self.clusters.cleanup(force=True)

    def cleanup(self, pipeline=None):
        if False:
            return 10
        'Cleans up cached states for the given pipeline. Noop if the given\n    pipeline is absent from the environment. Cleans up for all pipelines\n    if no pipeline is specified.'
        if pipeline:
            self.cleanup_pipeline(pipeline)
        else:
            self.cleanup_environment()
        self.evict_recording_manager(pipeline)
        self.evict_background_caching_job(pipeline)
        self.evict_test_stream_service_controller(pipeline)
        self.evict_computed_pcollections(pipeline)
        self.evict_cached_source_signature(pipeline)
        self.evict_pipeline_result(pipeline)
        self.evict_tracked_pipelines(pipeline)

    def _track_user_pipelines(self, watchable):
        if False:
            while True:
                i = 10
        'Tracks user pipelines from the given watchable.'
        pipelines = set()
        if isinstance(watchable, beam.Pipeline):
            pipelines.add(watchable)
        elif isinstance(watchable, dict):
            for v in watchable.values():
                if isinstance(v, beam.Pipeline):
                    pipelines.add(v)
        elif isinstance(watchable, Iterable):
            for v in watchable:
                if isinstance(v, beam.Pipeline):
                    pipelines.add(v)
        for p in pipelines:
            self._tracked_user_pipelines.add_user_pipeline(p)
            _ = self.get_cache_manager(p, create_if_absent=True)
            _ = self.get_recording_manager(p, create_if_absent=True)

    def watch(self, watchable):
        if False:
            while True:
                i = 10
        "Watches a watchable.\n\n    A watchable can be a dictionary of variable metadata such as locals(), a str\n    name of a module, a module object or an instance of a class. The variable\n    can come from any scope even local. Duplicated variable naming doesn't\n    matter since they are different instances. Duplicated variables are also\n    allowed when watching.\n    "
        if isinstance(watchable, dict):
            self._watching_dict_list.append(watchable.items())
        else:
            self._watching_set.add(watchable)
        self._track_user_pipelines(watchable)

    def watching(self):
        if False:
            i = 10
            return i + 15
        'Analyzes and returns a list of pair lists referring to variable names and\n    values from watched scopes.\n\n    Each entry in the list represents the variable defined within a watched\n    watchable. Currently, each entry holds a list of pairs. The format might\n    change in the future to hold more metadata. Duplicated pairs are allowed.\n    And multiple paris can have the same variable name as the "first" while\n    having different variable values as the "second" since variables in\n    different scopes can have the same name.\n    '
        watching = list(self._watching_dict_list)
        for watchable in self._watching_set:
            if isinstance(watchable, str):
                module = importlib.import_module(watchable)
                watching.append(vars(module).items())
            else:
                watching.append(vars(watchable).items())
        return watching

    def set_cache_manager(self, cache_manager, pipeline):
        if False:
            while True:
                i = 10
        'Sets the cache manager held by current Interactive Environment for the\n    given pipeline.'
        if self.get_cache_manager(pipeline) is cache_manager:
            return
        if self.get_cache_manager(pipeline):
            self.cleanup(pipeline)
        self._cache_managers[str(id(pipeline))] = cache_manager

    def get_cache_manager(self, pipeline, create_if_absent=False):
        if False:
            for i in range(10):
                print('nop')
        'Gets the cache manager held by current Interactive Environment for the\n    given pipeline. If the pipeline is absent from the environment while\n    create_if_absent is True, creates and returns a new file based cache\n    manager for the pipeline.'
        cache_manager = self._cache_managers.get(str(id(pipeline)), None)
        pipeline_runner = detect_pipeline_runner(pipeline)
        if not cache_manager and create_if_absent:
            cache_root = self.options.cache_root
            if cache_root:
                if cache_root.startswith('gs://'):
                    cache_dir = self._get_gcs_cache_dir(pipeline, cache_root)
                else:
                    cache_dir = tempfile.mkdtemp(dir=cache_root)
                    if not isinstance(pipeline_runner, direct_runner.DirectRunner):
                        _LOGGER.warning('A local cache directory has been specified while not using DirectRunner. It is recommended to cache into a GCS bucket instead.')
            else:
                staging_location = pipeline.options.get_all_options()['staging_location']
                if isinstance(pipeline_runner, DataflowRunner) and staging_location:
                    cache_dir = self._get_gcs_cache_dir(pipeline, staging_location)
                    _LOGGER.info('No cache_root detected. Defaulting to staging_location %s for cache location.', staging_location)
                else:
                    cache_dir = tempfile.mkdtemp(suffix=str(id(pipeline)), prefix='it-', dir=os.environ.get('TEST_TMPDIR', None))
            cache_manager = cache.FileBasedCacheManager(cache_dir)
            self._cache_managers[str(id(pipeline))] = cache_manager
        return cache_manager

    def evict_cache_manager(self, pipeline=None):
        if False:
            return 10
        'Evicts the cache manager held by current Interactive Environment for the\n    given pipeline. Noop if the pipeline is absent from the environment. If no\n    pipeline is specified, evicts for all pipelines.'
        self.cleanup(pipeline)
        if pipeline:
            return self._cache_managers.pop(str(id(pipeline)), None)
        self._cache_managers.clear()

    def set_recording_manager(self, recording_manager, pipeline):
        if False:
            while True:
                i = 10
        'Sets the recording manager for the given pipeline.'
        if self.get_recording_manager(pipeline) is recording_manager:
            return
        self._recording_managers[str(id(pipeline))] = recording_manager

    def get_recording_manager(self, pipeline, create_if_absent=False):
        if False:
            for i in range(10):
                print('nop')
        'Gets the recording manager for the given pipeline.'
        recording_manager = self._recording_managers.get(str(id(pipeline)), None)
        if not recording_manager and create_if_absent:
            pipeline_var = ''
            for w in self.watching():
                for (var, val) in w:
                    if val is pipeline:
                        pipeline_var = var
                        break
            recording_manager = RecordingManager(pipeline, pipeline_var)
            self._recording_managers[str(id(pipeline))] = recording_manager
        return recording_manager

    def evict_recording_manager(self, pipeline):
        if False:
            while True:
                i = 10
        'Evicts the recording manager for the given pipeline.\n\n    This stops the background caching job and clears the cache.\n    Noop if the pipeline is absent from the environment. If no\n    pipeline is specified, evicts for all pipelines.\n    '
        if not pipeline:
            for rm in self._recording_managers.values():
                rm.cancel()
                rm.clear()
            self._recording_managers = {}
            return
        recording_manager = self.get_recording_manager(pipeline)
        if recording_manager:
            recording_manager.cancel()
            recording_manager.clear()
            del self._recording_managers[str(id(pipeline))]

    def describe_all_recordings(self):
        if False:
            i = 10
            return i + 15
        'Returns a description of the recording for all watched pipelnes.'
        return {self.pipeline_id_to_pipeline(pid): rm.describe() for (pid, rm) in self._recording_managers.items()}

    def set_pipeline_result(self, pipeline, result):
        if False:
            while True:
                i = 10
        'Sets the pipeline run result. Adds one if absent. Otherwise, replace.'
        assert issubclass(type(pipeline), beam.Pipeline), 'pipeline must be an instance of apache_beam.Pipeline or its subclass'
        assert issubclass(type(result), runner.PipelineResult), 'result must be an instance of apache_beam.runners.runner.PipelineResult or its subclass'
        self._main_pipeline_results[str(id(pipeline))] = result

    def evict_pipeline_result(self, pipeline=None):
        if False:
            print('Hello World!')
        'Evicts the last run result of the given pipeline. Noop if the pipeline\n    is absent from the environment. If no pipeline is specified, evicts for all\n    pipelines.'
        if pipeline:
            return self._main_pipeline_results.pop(str(id(pipeline)), None)
        self._main_pipeline_results.clear()

    def pipeline_result(self, pipeline):
        if False:
            i = 10
            return i + 15
        'Gets the pipeline run result. None if absent.'
        return self._main_pipeline_results.get(str(id(pipeline)), None)

    def set_background_caching_job(self, pipeline, background_caching_job):
        if False:
            i = 10
            return i + 15
        'Sets the background caching job started from the given pipeline.'
        assert issubclass(type(pipeline), beam.Pipeline), 'pipeline must be an instance of apache_beam.Pipeline or its subclass'
        from apache_beam.runners.interactive.background_caching_job import BackgroundCachingJob
        assert isinstance(background_caching_job, BackgroundCachingJob), 'background_caching job must be an instance of BackgroundCachingJob'
        self._background_caching_jobs[str(id(pipeline))] = background_caching_job

    def get_background_caching_job(self, pipeline):
        if False:
            print('Hello World!')
        'Gets the background caching job started from the given pipeline.'
        return self._background_caching_jobs.get(str(id(pipeline)), None)

    def evict_background_caching_job(self, pipeline=None):
        if False:
            while True:
                i = 10
        'Evicts the background caching job started from the given pipeline. Noop\n    if the given pipeline is absent from the environment. If no pipeline is\n    specified, evicts for all pipelines.'
        if pipeline:
            return self._background_caching_jobs.pop(str(id(pipeline)), None)
        self._background_caching_jobs.clear()

    def set_test_stream_service_controller(self, pipeline, controller):
        if False:
            while True:
                i = 10
        'Sets the test stream service controller that has started a gRPC server\n    serving the test stream for any job started from the given user defined\n    pipeline.\n    '
        self._test_stream_service_controllers[str(id(pipeline))] = controller

    def get_test_stream_service_controller(self, pipeline):
        if False:
            print('Hello World!')
        'Gets the test stream service controller that has started a gRPC server\n    serving the test stream for any job started from the given user defined\n    pipeline.\n    '
        return self._test_stream_service_controllers.get(str(id(pipeline)), None)

    def evict_test_stream_service_controller(self, pipeline):
        if False:
            i = 10
            return i + 15
        'Evicts and pops the test stream service controller that has started a\n    gRPC server serving the test stream for any job started from the given\n    user defined pipeline. Noop if the given pipeline is absent from the\n    environment. If no pipeline is specified, evicts for all pipelines.\n    '
        if pipeline:
            return self._test_stream_service_controllers.pop(str(id(pipeline)), None)
        self._test_stream_service_controllers.clear()

    def is_terminated(self, pipeline):
        if False:
            for i in range(10):
                print('nop')
        'Queries if the most recent job (by executing the given pipeline) state\n    is in a terminal state. True if absent.'
        result = self.pipeline_result(pipeline)
        if result:
            return runner.PipelineState.is_terminal(result.state)
        return True

    def set_cached_source_signature(self, pipeline, signature):
        if False:
            i = 10
            return i + 15
        self._cached_source_signature[str(id(pipeline))] = signature

    def get_cached_source_signature(self, pipeline):
        if False:
            return 10
        return self._cached_source_signature.get(str(id(pipeline)), set())

    def evict_cached_source_signature(self, pipeline=None):
        if False:
            for i in range(10):
                print('nop')
        'Evicts the signature generated for each recorded source of the given\n    pipeline. Noop if the given pipeline is absent from the environment. If no\n    pipeline is specified, evicts for all pipelines.'
        if pipeline:
            return self._cached_source_signature.pop(str(id(pipeline)), None)
        self._cached_source_signature.clear()

    def track_user_pipelines(self):
        if False:
            print('Hello World!')
        'Record references to all user defined pipeline instances watched in\n    current environment.\n\n    Current static global singleton interactive environment holds references to\n    a set of pipeline instances defined by the user in the watched scope.\n    Interactive Beam features could use the references to determine if a given\n    pipeline is defined by user or implicitly created by Beam SDK or runners,\n    then handle them differently.\n\n    This is invoked every time a PTransform is to be applied if the current\n    code execution is under ipython due to the possibility that any user defined\n    pipeline can be re-evaluated through notebook cell re-execution at any time.\n\n    Each time this is invoked, it will check if there is a cache manager\n    already created for each user defined pipeline. If not, create one for it.\n\n    If a pipeline is no longer watched due to re-execution while its\n    PCollections are still in watched scope, the pipeline becomes anonymous but\n    still accessible indirectly through references to its PCollections. This\n    function also clears up internal states for those anonymous pipelines once\n    all their PCollections are anonymous.\n    '
        for watching in self.watching():
            for (_, val) in watching:
                if isinstance(val, beam.pipeline.Pipeline):
                    self._tracked_user_pipelines.add_user_pipeline(val)
                    _ = self.get_cache_manager(val, create_if_absent=True)
                    _ = self.get_recording_manager(val, create_if_absent=True)
        all_tracked_pipeline_ids = set(self._background_caching_jobs.keys()).union(set(self._test_stream_service_controllers.keys()), set(self._cache_managers.keys()), {str(id(pcoll.pipeline)) for pcoll in self._computed_pcolls}, set(self._cached_source_signature.keys()), set(self._main_pipeline_results.keys()))
        inspectable_pipelines = self._inspector.inspectable_pipelines
        for pipeline in all_tracked_pipeline_ids:
            if pipeline not in inspectable_pipelines:
                self.cleanup(pipeline)

    @property
    def tracked_user_pipelines(self):
        if False:
            return 10
        'Returns the user pipelines in this environment.'
        for p in self._tracked_user_pipelines:
            yield p

    def user_pipeline(self, derived_pipeline):
        if False:
            for i in range(10):
                print('nop')
        'Returns the user pipeline for the given derived pipeline.'
        return self._tracked_user_pipelines.get_user_pipeline(derived_pipeline)

    def add_user_pipeline(self, user_pipeline):
        if False:
            i = 10
            return i + 15
        self._tracked_user_pipelines.add_user_pipeline(user_pipeline)

    def add_derived_pipeline(self, user_pipeline, derived_pipeline):
        if False:
            print('Hello World!')
        'Adds the derived pipeline to the parent user pipeline.'
        self._tracked_user_pipelines.add_derived_pipeline(user_pipeline, derived_pipeline)

    def evict_tracked_pipelines(self, user_pipeline):
        if False:
            while True:
                i = 10
        'Evicts the user pipeline and its derived pipelines.'
        if user_pipeline:
            self._tracked_user_pipelines.evict(user_pipeline)
        else:
            self._tracked_user_pipelines.clear()

    def pipeline_id_to_pipeline(self, pid):
        if False:
            for i in range(10):
                print('nop')
        'Converts a pipeline id to a user pipeline.\n    '
        return self._tracked_user_pipelines.get_pipeline(pid)

    def mark_pcollection_computed(self, pcolls):
        if False:
            return 10
        'Marks computation completeness for the given pcolls.\n\n    Interactive Beam can use this information to determine if a computation is\n    needed to introspect the data of any given PCollection.\n    '
        self._computed_pcolls.update((pcoll for pcoll in pcolls))

    def evict_computed_pcollections(self, pipeline=None):
        if False:
            for i in range(10):
                print('nop')
        'Evicts all computed PCollections for the given pipeline. If no pipeline\n    is specified, evicts for all pipelines.\n    '
        if pipeline:
            discarded = set()
            for pcoll in self._computed_pcolls:
                if pcoll.pipeline is pipeline:
                    discarded.add(pcoll)
            self._computed_pcolls -= discarded
        else:
            self._computed_pcolls = set()

    @property
    def computed_pcollections(self):
        if False:
            for i in range(10):
                print('nop')
        return self._computed_pcolls

    def load_jquery_with_datatable(self):
        if False:
            while True:
                i = 10
        'Loads common resources to enable jquery with datatable configured for\n    notebook frontends if necessary. If the resources have been loaded, NOOP.\n\n    A window.interactive_beam_jquery with datatable plugin configured can be\n    used in following notebook cells once this is invoked.\n\n    #. There should only be one jQuery imported.\n    #. Datatable needs to be imported after jQuery is loaded.\n    #. Imported jQuery is attached to window named as jquery[version].\n    #. The window attachment needs to happen at the end of import chain until\n       all jQuery plugins are set.\n    '
        try:
            from IPython.display import Javascript
            from IPython.display import display_javascript
            display_javascript(Javascript(_JQUERY_WITH_DATATABLE_TEMPLATE.format(customized_script='')))
        except ImportError:
            pass

    def import_html_to_head(self, html_hrefs):
        if False:
            i = 10
            return i + 15
        "Imports given external HTMLs (supported through webcomponents) into\n    the head of the document.\n\n    On load of webcomponentsjs, import given HTMLs. If HTML import is already\n    supported, skip loading webcomponentsjs.\n\n    No matter how many times an HTML import occurs in the document, only the\n    first occurrence really embeds the external HTML. In a notebook environment,\n    the body of the document is always changing due to cell [re-]execution,\n    deletion and re-ordering. Thus, HTML imports shouldn't be put in the body\n    especially the output areas of notebook cells.\n    "
        try:
            from IPython.display import Javascript
            from IPython.display import display_javascript
            display_javascript(Javascript(_HTML_IMPORT_TEMPLATE.format(hrefs=html_hrefs)))
        except ImportError:
            pass

    def get_sql_chain(self, pipeline, set_user_pipeline=False):
        if False:
            for i in range(10):
                print('nop')
        if pipeline not in self.sql_chain:
            self.sql_chain[pipeline] = SqlChain()
        chain = self.sql_chain[pipeline]
        if set_user_pipeline:
            if chain.user_pipeline and chain.user_pipeline is not pipeline:
                raise ValueError('The beam_sql magic tries to query PCollections from multiple pipelines: %s and %s', chain.user_pipeline, pipeline)
            chain.user_pipeline = pipeline
        return chain

    def _get_gcs_cache_dir(self, pipeline, cache_dir):
        if False:
            return 10
        cache_dir_path = PurePath(cache_dir)
        if len(cache_dir_path.parts) < 2:
            _LOGGER.error('GCS bucket cache path "%s" is too short to be valid. See https://cloud.google.com/storage/docs/naming-buckets for the expected format.', cache_dir)
            raise ValueError('cache_root GCS bucket path is invalid.')
        bucket_name = cache_dir_path.parts[1]
        assert_bucket_exists(bucket_name)
        return 'gs://{}/{}'.format('/'.join(cache_dir_path.parts[1:]), id(pipeline))