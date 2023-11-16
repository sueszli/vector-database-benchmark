"""Module of Interactive Beam features that can be used in notebook.

The purpose of the module is to reduce the learning curve of Interactive Beam
users, provide a single place for importing and add sugar syntax for all
Interactive Beam components. It gives users capability to interact with existing
environment/session/context for Interactive Beam and visualize PCollections as
bounded dataset. In the meantime, it hides the interactivity implementation
from users so that users can focus on developing Beam pipeline without worrying
about how hidden states in the interactive session are managed.

A convention to import this module:
  from apache_beam.runners.interactive import interactive_beam as ib

Note: If you want backward-compatibility, only invoke interfaces provided by
this module in your notebook or application code.
"""
import logging
from datetime import timedelta
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
import pandas as pd
import apache_beam as beam
from apache_beam.dataframe.frame_base import DeferredBase
from apache_beam.options.pipeline_options import FlinkRunnerOptions
from apache_beam.runners.interactive import interactive_environment as ie
from apache_beam.runners.interactive.dataproc.dataproc_cluster_manager import DataprocClusterManager
from apache_beam.runners.interactive.dataproc.types import ClusterIdentifier
from apache_beam.runners.interactive.dataproc.types import ClusterMetadata
from apache_beam.runners.interactive.display import pipeline_graph
from apache_beam.runners.interactive.display.pcoll_visualization import visualize
from apache_beam.runners.interactive.display.pcoll_visualization import visualize_computed_pcoll
from apache_beam.runners.interactive.options import interactive_options
from apache_beam.runners.interactive.utils import deferred_df_to_pcollection
from apache_beam.runners.interactive.utils import elements_to_df
from apache_beam.runners.interactive.utils import find_pcoll_name
from apache_beam.runners.interactive.utils import progress_indicated
from apache_beam.runners.runner import PipelineState
_LOGGER = logging.getLogger(__name__)

class Options(interactive_options.InteractiveOptions):
    """Options that guide how Interactive Beam works."""

    @property
    def enable_recording_replay(self):
        if False:
            return 10
        'Whether replayable source data recorded should be replayed for multiple\n    PCollection evaluations and pipeline runs as long as the data recorded is\n    still valid.'
        return self.capture_control._enable_capture_replay

    @enable_recording_replay.setter
    def enable_recording_replay(self, value):
        if False:
            while True:
                i = 10
        'Sets whether source data recorded should be replayed. True - Enables\n    recording of replayable source data so that following PCollection\n    evaluations and pipeline runs always use the same data recorded;\n    False - Disables recording of replayable source data so that following\n    PCollection evaluation and pipeline runs always use new data from sources.\n    '
        _ = ie.current_env()
        if value:
            _LOGGER.info('Record replay is enabled. When a PCollection is evaluated or the pipeline is executed, existing data recorded from previous computations will be replayed for consistent results. If no recorded data is available, new data from recordable sources will be recorded.')
        else:
            _LOGGER.info('Record replay is disabled. The next time a PCollection is evaluated or the pipeline is executed, new data will always be consumed from sources in the pipeline. You will not have replayability until re-enabling this option.')
        self.capture_control._enable_capture_replay = value

    @property
    def recordable_sources(self):
        if False:
            return 10
        'Interactive Beam automatically records data from sources in this set.\n    '
        return self.capture_control._capturable_sources

    @property
    def recording_duration(self):
        if False:
            print('Hello World!')
        'The data recording of sources ends as soon as the background source\n    recording job has run for this long.'
        return self.capture_control._capture_duration

    @recording_duration.setter
    def recording_duration(self, value):
        if False:
            i = 10
            return i + 15
        "Sets the recording duration as a timedelta. The input can be a\n    datetime.timedelta, a possitive integer as seconds or a string\n    representation that is parsable by pandas.to_timedelta.\n\n    Example::\n\n      # Sets the recording duration limit to 10 seconds.\n      ib.options.recording_duration = timedelta(seconds=10)\n      ib.options.recording_duration = 10\n      ib.options.recording_duration = '10s'\n      # Explicitly control the recordings.\n      ib.recordings.stop(p)\n      ib.recordings.clear(p)\n      ib.recordings.record(p)\n      # The next PCollection evaluation uses fresh data from sources,\n      # and the data recorded will be replayed until another clear.\n      ib.collect(some_pcoll)\n    "
        duration = None
        if isinstance(value, int):
            assert value > 0, 'Duration must be a positive value.'
            duration = timedelta(seconds=value)
        elif isinstance(value, str):
            duration = pd.to_timedelta(value)
        else:
            assert isinstance(value, timedelta), 'The input can only abe a datetime.timedelta, a possitive integer as seconds, or a string representation that is parsable by pandas.to_timedelta.'
            duration = value
        if self.capture_control._capture_duration.total_seconds() != duration.total_seconds():
            _ = ie.current_env()
            _LOGGER.info('You have changed recording duration from %s seconds to %s seconds. To allow new data to be recorded for the updated duration the next time a PCollection is evaluated or the pipeline is executed, please invoke ib.recordings.stop, ib.recordings.clear and ib.recordings.record.', self.capture_control._capture_duration.total_seconds(), duration.total_seconds())
            self.capture_control._capture_duration = duration

    @property
    def recording_size_limit(self):
        if False:
            return 10
        'The data recording of sources ends as soon as the size (in bytes) of data\n    recorded from recordable sources reaches the limit.'
        return self.capture_control._capture_size_limit

    @recording_size_limit.setter
    def recording_size_limit(self, value):
        if False:
            i = 10
            return i + 15
        'Sets the recording size in bytes.\n\n    Example::\n\n      # Sets the recording size limit to 1GB.\n      interactive_beam.options.recording_size_limit = 1e9\n    '
        if self.capture_control._capture_size_limit != value:
            _ = ie.current_env()
            _LOGGER.info('You have changed recording size limit from %s bytes to %s bytes. To allow new data to be recorded under the updated size limit the next time a PCollection is recorded or the pipeline is executed, please invoke ib.recordings.stop, ib.recordings.clear and ib.recordings.record.', self.capture_control._capture_size_limit, value)
            self.capture_control._capture_size_limit = value

    @property
    def display_timestamp_format(self):
        if False:
            while True:
                i = 10
        "The format in which timestamps are displayed.\n\n    Default is '%Y-%m-%d %H:%M:%S.%f%z', e.g. 2020-02-01 15:05:06.000015-08:00.\n    "
        return self._display_timestamp_format

    @display_timestamp_format.setter
    def display_timestamp_format(self, value):
        if False:
            print('Hello World!')
        "Sets the format in which timestamps are displayed.\n\n    Default is '%Y-%m-%d %H:%M:%S.%f%z', e.g. 2020-02-01 15:05:06.000015-08:00.\n\n    Example::\n\n      # Sets the format to not display the timezone or microseconds.\n      interactive_beam.options.display_timestamp_format = %Y-%m-%d %H:%M:%S'\n    "
        self._display_timestamp_format = value

    @property
    def display_timezone(self):
        if False:
            while True:
                i = 10
        'The timezone in which timestamps are displayed.\n\n    Defaults to local timezone.\n    '
        return self._display_timezone

    @display_timezone.setter
    def display_timezone(self, value):
        if False:
            return 10
        "Sets the timezone (datetime.tzinfo) in which timestamps are displayed.\n\n    Defaults to local timezone.\n\n    Example::\n\n      # Imports the timezone library.\n      from pytz import timezone\n\n      # Will display all timestamps in the US/Eastern time zone.\n      tz = timezone('US/Eastern')\n\n      # You can also use dateutil.tz to get a timezone.\n      tz = dateutil.tz.gettz('US/Eastern')\n\n      interactive_beam.options.display_timezone = tz\n    "
        self._display_timezone = value

    @property
    def cache_root(self):
        if False:
            for i in range(10):
                print('nop')
        'The cache directory specified by the user.\n\n    Defaults to None.\n    '
        return self._cache_root

    @cache_root.setter
    def cache_root(self, value):
        if False:
            return 10
        "Sets the cache directory.\n\n    Defaults to None.\n\n    Example of local directory usage::\n      interactive_beam.options.cache_root = '/Users/username/my/cache/dir'\n\n    Example of GCS directory usage::\n      interactive_beam.options.cache_root = 'gs://my-gcs-bucket/cache/dir'\n    "
        _LOGGER.warning('Interactive Beam has detected a set value for the cache_root option. Please note: existing cache managers will not have their current cache directory changed. The option must be set in Interactive Beam prior to the initialization of new pipelines to take effect. To apply changes to new pipelines, the kernel must be restarted or the pipeline creation codes must be re-executed. ')
        self._cache_root = value

class Recordings:
    """An introspection interface for recordings for pipelines.

  When a user materializes a PCollection onto disk (eg. ib.show) for a streaming
  pipeline, a background source recording job is started. This job pulls data
  from all defined unbounded sources for that PCollection's pipeline. The
  following methods allow for introspection into that background recording job.
  """

    def describe(self, pipeline=None):
        if False:
            print('Hello World!')
        'Returns a description of all the recordings for the given pipeline.\n\n    If no pipeline is given then this returns a dictionary of descriptions for\n    all pipelines.\n    '
        if pipeline:
            ie.current_env().get_recording_manager(pipeline, create_if_absent=True)
        description = ie.current_env().describe_all_recordings()
        if pipeline:
            return description[pipeline]
        return description

    def clear(self, pipeline):
        if False:
            while True:
                i = 10
        'Clears all recordings of the given pipeline. Returns True if cleared.'
        description = self.describe(pipeline)
        if not PipelineState.is_terminal(description['state']) and description['state'] != PipelineState.STOPPED:
            _LOGGER.warning('Trying to clear a recording with a running pipeline. Did you forget to call ib.recordings.stop?')
            return False
        ie.current_env().cleanup(pipeline)
        return True

    def stop(self, pipeline):
        if False:
            while True:
                i = 10
        'Stops the background source recording of the given pipeline.'
        recording_manager = ie.current_env().get_recording_manager(pipeline, create_if_absent=True)
        recording_manager.cancel()

    def record(self, pipeline):
        if False:
            for i in range(10):
                print('nop')
        'Starts a background source recording job for the given pipeline. Returns\n    True if the recording job was started.\n    '
        description = self.describe(pipeline)
        if not PipelineState.is_terminal(description['state']) and description['state'] != PipelineState.STOPPED:
            _LOGGER.warning('Trying to start a recording with a running pipeline. Did you forget to call ib.recordings.stop?')
            return False
        if description['size'] > 0:
            _LOGGER.warning('A recording already exists for this pipeline. To start a recording, make sure to call ib.recordings.clear first.')
            return False
        recording_manager = ie.current_env().get_recording_manager(pipeline, create_if_absent=True)
        return recording_manager.record_pipeline()

class Clusters:
    """An interface to control clusters implicitly created and managed by
  the current interactive environment. This class is not needed and
  should not be used otherwise.

  Do not use it for clusters a user explicitly manages: e.g., if you have
  a Flink cluster running somewhere and provides the flink master when
  running a pipeline with the FlinkRunner, the cluster will not be tracked
  or managed by Beam.
  To reuse the same cluster for your pipelines, use the same pipeline
  options: e.g., a pipeline option with the same flink master if you are
  using FlinkRunner.

  This module is experimental. No backwards-compatibility guarantees.

  Interactive Beam automatically creates/reuses existing worker clusters to
  execute pipelines when it detects the need from configurations.
  Currently, the only supported cluster implementation is Flink running on
  Cloud Dataproc.

  To configure a pipeline to run on Cloud Dataproc with Flink, set the
  underlying runner of the InteractiveRunner to FlinkRunner and the pipeline
  options to indicate where on Cloud the FlinkRunner should be deployed to.

    An example to enable automatic Dataproc cluster creation/reuse::

      options = PipelineOptions([
          '--project=my-project',
          '--region=my-region',
          '--environment_type=DOCKER'])
      pipeline = beam.Pipeline(InteractiveRunner(
          underlying_runner=FlinkRunner()), options=options)

  Reuse a pipeline options in another pipeline would configure Interactive Beam
  to reuse the same Dataproc cluster implicitly managed by the current
  interactive environment.
  If a flink_master is identified as a known cluster, the corresponding cluster
  is also resued.
  Furthermore, if a cluster is explicitly created by using a pipeline as an
  identifier to a known cluster, the cluster is reused.

    An example::

      # If pipeline runs on a known cluster, below code reuses the cluster
      # manager without creating a new one.
      dcm = ib.clusters.create(pipeline)

  To provision the cluster, use WorkerOptions. Supported configurations are::

    1. subnetwork
    2. num_workers
    3. machine_type

  To configure a pipeline to run on an existing FlinkRunner deployed elsewhere,
  set the flink_master explicitly so no cluster will be created/reused.

    An example pipeline options to skip automatic Dataproc cluster usage::

      options = PipelineOptions([
          '--flink_master=some.self.hosted.flink:port',
          '--environment_type=DOCKER'])

  To configure a pipeline to run on a local FlinkRunner, explicitly set the
  default cluster metadata to None: ib.clusters.set_default_cluster(None).
  """
    DATAPROC_FLINK_VERSION = '1.12'
    DATAPROC_MINIMUM_WORKER_NUM = 2

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        self.dataproc_cluster_managers: Dict[ClusterMetadata, DataprocClusterManager] = {}
        self.master_urls: Dict[str, ClusterMetadata] = {}
        self.pipelines: Dict[beam.Pipeline, DataprocClusterManager] = {}
        self.default_cluster_metadata: Optional[ClusterMetadata] = None

    def create(self, cluster_identifier: ClusterIdentifier) -> DataprocClusterManager:
        if False:
            return 10
        'Creates a Dataproc cluster manager provisioned for the cluster\n    identified. If the cluster is known, returns an existing cluster manager.\n    '
        cluster_metadata = self.cluster_metadata(cluster_identifier)
        if not cluster_metadata:
            raise ValueError('Unknown cluster identifier: %s. Cannot create or reusea Dataproc cluster.')
        if not cluster_metadata.region:
            _LOGGER.info('No region information was detected, defaulting Dataproc cluster region to: us-central1.')
            cluster_metadata.region = 'us-central1'
        elif cluster_metadata.region == 'global':
            raise ValueError('Clusters in the global region are not supported.')
        if cluster_metadata.num_workers and cluster_metadata.num_workers < self.DATAPROC_MINIMUM_WORKER_NUM:
            _LOGGER.info('At least %s workers are required for a cluster, defaulting to %s.', self.DATAPROC_MINIMUM_WORKER_NUM, self.DATAPROC_MINIMUM_WORKER_NUM)
            cluster_metadata.num_workers = self.DATAPROC_MINIMUM_WORKER_NUM
        known_dcm = self.dataproc_cluster_managers.get(cluster_metadata, None)
        if known_dcm:
            return known_dcm
        dcm = DataprocClusterManager(cluster_metadata)
        dcm.create_flink_cluster()
        derived_meta = dcm.cluster_metadata
        self.dataproc_cluster_managers[derived_meta] = dcm
        self.master_urls[derived_meta.master_url] = derived_meta
        self.set_default_cluster(derived_meta)
        return dcm

    def cleanup(self, cluster_identifier: Optional[ClusterIdentifier]=None, force: bool=False) -> None:
        if False:
            return 10
        'Cleans up the cluster associated with the given cluster_identifier.\n\n    When None cluster_identifier is provided: if force is True, cleans up for\n    all clusters; otherwise, do a dry run and NOOP.\n    If a beam.Pipeline is given as the ClusterIdentifier while multiple\n    pipelines share the same cluster, it only cleans up the association between\n    the pipeline and the cluster identified.\n    If the cluster_identifier is unknown, NOOP.\n    '
        if not cluster_identifier:
            dcm_to_cleanup = set(self.dataproc_cluster_managers.values())
            if force:
                for dcm in dcm_to_cleanup:
                    self._cleanup(dcm)
                self.default_cluster_metadata = None
            else:
                _LOGGER.warning('No cluster_identifier provided. If you intend to clean up all clusters, invoke ib.clusters.cleanup(force=True). Current clusters are %s.', self.describe())
        elif isinstance(cluster_identifier, beam.Pipeline):
            p = cluster_identifier
            dcm = self.pipelines.pop(p, None)
            if dcm:
                dcm.pipelines.remove(p)
                p_flink_options = p.options.view_as(FlinkRunnerOptions)
                p_flink_options.flink_master = '[auto]'
                p_flink_options.flink_version = None
                if not dcm.pipelines:
                    self._cleanup(dcm)
        else:
            if isinstance(cluster_identifier, str):
                meta = self.master_urls.get(cluster_identifier, None)
            else:
                meta = cluster_identifier
            dcm = self.dataproc_cluster_managers.get(meta, None)
            if dcm:
                self._cleanup(dcm)

    def describe(self, cluster_identifier: Optional[ClusterIdentifier]=None) -> Union[ClusterMetadata, List[ClusterMetadata]]:
        if False:
            while True:
                i = 10
        'Describes the ClusterMetadata by a ClusterIdentifier.\n\n    If no cluster_identifier is given or if the cluster_identifier is unknown,\n    it returns descriptions for all known clusters.\n\n    Example usage:\n    # Describe the cluster executing work for a pipeline.\n    ib.clusters.describe(pipeline)\n    # Describe the cluster with the flink master url.\n    ib.clusters.describe(master_url)\n    # Describe all existing clusters.\n    ib.clusters.describe()\n    '
        if cluster_identifier:
            meta = self._cluster_metadata(cluster_identifier)
            if meta in self.dataproc_cluster_managers:
                return meta
        return list(self.dataproc_cluster_managers.keys())

    def set_default_cluster(self, cluster_identifier: Optional[ClusterIdentifier]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Temporarily sets the default metadata for creating or reusing a\n    DataprocClusterManager. It is always updated to the most recently created\n    cluster.\n\n    If no known ClusterMetadata can be identified by the ClusterIdentifer, NOOP.\n    If None is set, next time when Flink is in use, if no cluster is explicitly\n    configured by a pipeline, the job runs locally.\n    '
        if cluster_identifier:
            self.default_cluster_metadata = self.cluster_metadata(cluster_identifier)
        else:
            self.default_cluster_metadata = None

    def cluster_metadata(self, cluster_identifier: Optional[ClusterIdentifier]=None) -> Optional[ClusterMetadata]:
        if False:
            return 10
        'Fetches the ClusterMetadata by a ClusterIdentifier that could be a\n    URL in string, a Beam pipeline, or an equivalent to a known ClusterMetadata;\n\n    If the given cluster_identifier is an URL or a pipeline that is unknown to\n    the current environment, the default cluster metadata (could be None) is\n    returned.\n    If the given cluster_identifier is a ClusterMetadata but unknown to the\n    current environment, passes it through (NOOP).\n    '
        meta = self._cluster_metadata(cluster_identifier)
        return meta if meta else self.default_cluster_metadata

    def _cluster_metadata(self, cluster_identifier: Optional[ClusterIdentifier]=None) -> Optional[ClusterMetadata]:
        if False:
            for i in range(10):
                print('nop')
        meta = None
        if cluster_identifier:
            if isinstance(cluster_identifier, str):
                meta = self.master_urls.get(cluster_identifier, None)
            elif isinstance(cluster_identifier, beam.Pipeline):
                dcm = self.pipelines.get(cluster_identifier, None)
                if dcm:
                    meta = dcm.cluster_metadata
            elif isinstance(cluster_identifier, ClusterMetadata):
                meta = cluster_identifier
                if meta in self.dataproc_cluster_managers:
                    meta = self.dataproc_cluster_managers[meta].cluster_metadata
                elif meta and self.default_cluster_metadata and (meta.cluster_name == self.default_cluster_metadata.cluster_name):
                    _LOGGER.warning('Cannot change the configuration of the running cluster %s. Existing is %s, desired is %s.', self.default_cluster_metadata.cluster_name, self.default_cluster_metadata, meta)
                    meta.reset_name()
                    _LOGGER.warning('To avoid conflict, issuing a new cluster name %s for a new cluster.', meta.cluster_name)
            else:
                raise TypeError('A cluster_identifier should be Optional[Union[str, beam.Pipeline, ClusterMetadata], instead %s was given.', type(cluster_identifier))
        return meta

    def _cleanup(self, dcm: DataprocClusterManager) -> None:
        if False:
            return 10
        dcm.cleanup()
        self.dataproc_cluster_managers.pop(dcm.cluster_metadata, None)
        self.master_urls.pop(dcm.cluster_metadata.master_url, None)
        for p in dcm.pipelines:
            self.pipelines.pop(p, None)
        if dcm.cluster_metadata == self.default_cluster_metadata:
            self.default_cluster_metadata = None
options = Options()
recordings = Recordings()
clusters = Clusters()

def watch(watchable):
    if False:
        while True:
            i = 10
    "Monitors a watchable.\n\n  This allows Interactive Beam to implicitly pass on the information about the\n  location of your pipeline definition.\n\n  Current implementation mainly watches for PCollection variables defined in\n  user code. A watchable can be a dictionary of variable metadata such as\n  locals(), a str name of a module, a module object or an instance of a class.\n  The variable can come from any scope even local variables in a method of a\n  class defined in a module.\n\n    Below are all valid::\n\n      watch(__main__)  # if import __main__ is already invoked\n      watch('__main__')  # does not require invoking import __main__ beforehand\n      watch(self)  # inside a class\n      watch(SomeInstance())  # an instance of a class\n      watch(locals())  # inside a function, watching local variables within\n\n  If you write a Beam pipeline in the __main__ module directly, since the\n  __main__ module is always watched, you don't have to instruct Interactive\n  Beam. If your Beam pipeline is defined in some module other than __main__,\n  such as inside a class function or a unit test, you can watch() the scope.\n\n    For example::\n\n      class Foo(object)\n        def run_pipeline(self):\n          with beam.Pipeline() as p:\n            init_pcoll = p |  'Init Create' >> beam.Create(range(10))\n            watch(locals())\n          return init_pcoll\n      init_pcoll = Foo().run_pipeline()\n\n    Interactive Beam caches init_pcoll for the first run.\n\n    Then you can use::\n\n      show(init_pcoll)\n\n    To visualize data from init_pcoll once the pipeline is executed.\n  "
    ie.current_env().watch(watchable)

@progress_indicated
def show(*pcolls, include_window_info=False, visualize_data=False, n='inf', duration='inf'):
    if False:
        i = 10
        return i + 15
    "Shows given PCollections in an interactive exploratory way if used within\n  a notebook, or prints a heading sampled data if used within an ipython shell.\n  Noop if used in a non-interactive environment.\n\n  Args:\n    include_window_info: (optional) if True, windowing information of the\n        data will be visualized too. Default is false.\n    visualize_data: (optional) by default, the visualization contains data\n        tables rendering data from given pcolls separately as if they are\n        converted into dataframes. If visualize_data is True, there will be a\n        more dive-in widget and statistically overview widget of the data.\n        Otherwise, those 2 data visualization widgets will not be displayed.\n    n: (optional) max number of elements to visualize. Default 'inf'.\n    duration: (optional) max duration of elements to read in integer seconds or\n        a string duration. Default 'inf'.\n\n  The given pcolls can be dictionary of PCollections (as values), or iterable\n  of PCollections or plain PCollection values.\n\n  The user can specify either the max number of elements with `n` to read\n  or the maximum duration of elements to read with `duration`. When a limiter is\n  not supplied, it is assumed to be infinite.\n\n  By default, the visualization contains data tables rendering data from given\n  pcolls separately as if they are converted into dataframes. If visualize_data\n  is True, there will be a more dive-in widget and statistically overview widget\n  of the data. Otherwise, those 2 data visualization widgets will not be\n  displayed.\n\n  Ad hoc builds a pipeline fragment including only transforms that are\n  necessary to produce data for given PCollections pcolls, runs the pipeline\n  fragment to compute data for those pcolls and then visualizes the data.\n\n  The function is always blocking. If used within a notebook, the data\n  visualized might be dynamically updated before the function returns as more\n  and more data could getting processed and emitted when the pipeline fragment\n  is being executed. If used within an ipython shell, there will be no dynamic\n  plotting but a static plotting in the end of pipeline fragment execution.\n\n  The PCollections given must belong to the same pipeline.\n\n    For example::\n\n      p = beam.Pipeline(InteractiveRunner())\n      init = p | 'Init' >> beam.Create(range(1000))\n      square = init | 'Square' >> beam.Map(lambda x: x * x)\n      cube = init | 'Cube' >> beam.Map(lambda x: x ** 3)\n\n      # Below builds a pipeline fragment from the defined pipeline `p` that\n      # contains only applied transforms of `Init` and `Square`. Then the\n      # interactive runner runs the pipeline fragment implicitly to compute data\n      # represented by PCollection `square` and visualizes it.\n      show(square)\n\n      # This is equivalent to `show(square)` because `square` depends on `init`\n      # and `init` is included in the pipeline fragment and computed anyway.\n      show(init, square)\n\n      # Below is similar to running `p.run()`. It computes data for both\n      # PCollection `square` and PCollection `cube`, then visualizes them.\n      show(square, cube)\n  "
    flatten_pcolls = []
    for pcoll_container in pcolls:
        if isinstance(pcoll_container, dict):
            flatten_pcolls.extend(pcoll_container.values())
        elif isinstance(pcoll_container, (beam.pvalue.PCollection, DeferredBase)):
            flatten_pcolls.append(pcoll_container)
        else:
            try:
                flatten_pcolls.extend(iter(pcoll_container))
            except TypeError:
                raise ValueError('The given pcoll %s is not a dict, an iterable or a PCollection.' % pcoll_container)
    pcolls = set()
    element_types = {}
    for pcoll in flatten_pcolls:
        if isinstance(pcoll, DeferredBase):
            (pcoll, element_type) = deferred_df_to_pcollection(pcoll)
            watch({'anonymous_pcollection_{}'.format(id(pcoll)): pcoll})
        else:
            element_type = pcoll.element_type
        element_types[pcoll] = element_type
        pcolls.add(pcoll)
        assert isinstance(pcoll, beam.pvalue.PCollection), '{} is not an apache_beam.pvalue.PCollection.'.format(pcoll)
    assert len(pcolls) > 0, 'Need at least 1 PCollection to show data visualization.'
    pcoll_pipeline = next(iter(pcolls)).pipeline
    user_pipeline = ie.current_env().user_pipeline(pcoll_pipeline)
    if not user_pipeline:
        watch({'anonymous_pipeline_{}'.format(id(pcoll_pipeline)): pcoll_pipeline})
        user_pipeline = pcoll_pipeline
    if isinstance(n, str):
        assert n == 'inf', "Currently only the string 'inf' is supported. This denotes reading elements until the recording is stopped via a kernel interrupt."
    elif isinstance(n, int):
        assert n > 0, "n needs to be positive or the string 'inf'"
    if isinstance(duration, int):
        assert duration > 0, "duration needs to be positive, a duration string, or the string 'inf'"
    if n == 'inf':
        n = float('inf')
    if duration == 'inf':
        duration = float('inf')
    previously_computed_pcolls = {pcoll for pcoll in pcolls if pcoll in ie.current_env().computed_pcollections}
    for pcoll in previously_computed_pcolls:
        visualize_computed_pcoll(find_pcoll_name(pcoll), pcoll, n, duration, include_window_info=include_window_info, display_facets=visualize_data)
    pcolls = pcolls - previously_computed_pcolls
    recording_manager = ie.current_env().get_recording_manager(user_pipeline, create_if_absent=True)
    recording = recording_manager.record(pcolls, max_n=n, max_duration=duration)
    try:
        if ie.current_env().is_in_notebook:
            for stream in recording.computed().values():
                visualize(stream, include_window_info=include_window_info, display_facets=visualize_data, element_type=element_types[stream.pcoll])
        elif ie.current_env().is_in_ipython:
            for stream in recording.computed().values():
                visualize(stream, include_window_info=include_window_info, element_type=element_types[stream.pcoll])
        if recording.is_computed():
            return
        if ie.current_env().is_in_notebook:
            for stream in recording.uncomputed().values():
                visualize(stream, dynamic_plotting_interval=1, include_window_info=include_window_info, display_facets=visualize_data, element_type=element_types[stream.pcoll])
        recording.wait_until_finish()
        if ie.current_env().is_in_ipython and (not ie.current_env().is_in_notebook):
            for stream in recording.computed().values():
                visualize(stream, include_window_info=include_window_info)
    except KeyboardInterrupt:
        if recording:
            recording.cancel()

@progress_indicated
def collect(pcoll, n='inf', duration='inf', include_window_info=False):
    if False:
        i = 10
        return i + 15
    "Materializes the elements from a PCollection into a Dataframe.\n\n  This reads each element from file and reads only the amount that it needs\n  into memory. The user can specify either the max number of elements to read\n  or the maximum duration of elements to read. When a limiter is not supplied,\n  it is assumed to be infinite.\n\n  Args:\n    n: (optional) max number of elements to visualize. Default 'inf'.\n    duration: (optional) max duration of elements to read in integer seconds or\n        a string duration. Default 'inf'.\n    include_window_info: (optional) if True, appends the windowing information\n        to each row. Default False.\n\n  For example::\n\n    p = beam.Pipeline(InteractiveRunner())\n    init = p | 'Init' >> beam.Create(range(10))\n    square = init | 'Square' >> beam.Map(lambda x: x * x)\n\n    # Run the pipeline and bring the PCollection into memory as a Dataframe.\n    in_memory_square = head(square, n=5)\n  "
    if isinstance(pcoll, DeferredBase):
        (pcoll, element_type) = deferred_df_to_pcollection(pcoll)
        watch({'anonymous_pcollection_{}'.format(id(pcoll)): pcoll})
    else:
        element_type = pcoll.element_type
    assert isinstance(pcoll, beam.pvalue.PCollection), '{} is not an apache_beam.pvalue.PCollection.'.format(pcoll)
    if isinstance(n, str):
        assert n == 'inf', "Currently only the string 'inf' is supported. This denotes reading elements until the recording is stopped via a kernel interrupt."
    elif isinstance(n, int):
        assert n > 0, "n needs to be positive or the string 'inf'"
    if isinstance(duration, int):
        assert duration > 0, "duration needs to be positive, a duration string, or the string 'inf'"
    if n == 'inf':
        n = float('inf')
    if duration == 'inf':
        duration = float('inf')
    user_pipeline = ie.current_env().user_pipeline(pcoll.pipeline)
    if not user_pipeline:
        watch({'anonymous_pipeline_{}'.format(id(pcoll.pipeline)): pcoll.pipeline})
        user_pipeline = pcoll.pipeline
    recording_manager = ie.current_env().get_recording_manager(user_pipeline, create_if_absent=True)
    if pcoll in ie.current_env().computed_pcollections:
        pcoll_name = find_pcoll_name(pcoll)
        elements = list(recording_manager.read(pcoll_name, pcoll, n, duration).read())
        return elements_to_df(elements, include_window_info=include_window_info, element_type=element_type)
    recording = recording_manager.record([pcoll], max_n=n, max_duration=duration)
    try:
        elements = list(recording.stream(pcoll).read())
    except KeyboardInterrupt:
        recording.cancel()
        return pd.DataFrame()
    if n == float('inf'):
        n = None
    return elements_to_df(elements, include_window_info=include_window_info, element_type=element_type)[:n]

@progress_indicated
def show_graph(pipeline):
    if False:
        while True:
            i = 10
    'Shows the current pipeline shape of a given Beam pipeline as a DAG.\n  '
    pipeline_graph.PipelineGraph(pipeline).display_graph()

def evict_recorded_data(pipeline=None):
    if False:
        return 10
    'Forcefully evicts all recorded replayable data for the given pipeline. If\n  no pipeline is specified, evicts for all user defined pipelines.\n\n  Once invoked, Interactive Beam will record new data based on the guidance of\n  options the next time it evaluates/visualizes PCollections or runs pipelines.\n  '
    from apache_beam.runners.interactive.options import capture_control
    capture_control.evict_captured_data(pipeline)