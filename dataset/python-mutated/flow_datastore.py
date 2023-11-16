import itertools
import json
from .. import metaflow_config
from .content_addressed_store import ContentAddressedStore
from .task_datastore import TaskDataStore

class FlowDataStore(object):
    default_storage_impl = None

    def __init__(self, flow_name, environment, metadata=None, event_logger=None, monitor=None, storage_impl=None, ds_root=None):
        if False:
            return 10
        '\n        Initialize a Flow level datastore.\n\n        This datastore can then be used to get TaskDataStore to store artifacts\n        and metadata about a task as well as a ContentAddressedStore to store\n        things like packages, etc.\n\n        Parameters\n        ----------\n        flow_name : str\n            The name of the flow\n        environment : MetaflowEnvironment\n            Environment this datastore is operating in\n        metadata : MetadataProvider, optional\n            The metadata provider to use and update if needed, by default None\n        event_logger : EventLogger, optional\n            EventLogger to use to report events, by default None\n        monitor : Monitor, optional\n            Monitor to use to measure/monitor events, by default None\n        storage_impl : type\n            Class for the backing DataStoreStorage to use; if not provided use\n            default_storage_impl, optional\n        ds_root : str\n            The optional root for this datastore; if not provided, use the\n            default for the DataStoreStorage, optional\n        '
        storage_impl = storage_impl if storage_impl else self.default_storage_impl
        if storage_impl is None:
            raise RuntimeError('No datastore storage implementation specified')
        self._storage_impl = storage_impl(ds_root)
        self.TYPE = self._storage_impl.TYPE
        self.flow_name = flow_name
        self.environment = environment
        self.metadata = metadata
        self.logger = event_logger
        self.monitor = monitor
        self.ca_store = ContentAddressedStore(self._storage_impl.path_join(self.flow_name, 'data'), self._storage_impl)

    @property
    def datastore_root(self):
        if False:
            return 10
        return self._storage_impl.datastore_root

    def get_latest_task_datastores(self, run_id=None, steps=None, pathspecs=None, allow_not_done=False):
        if False:
            return 10
        "\n        Return a list of TaskDataStore for a subset of the tasks.\n\n        We filter the list based on `steps` if non-None.\n        Alternatively, `pathspecs` can contain the exact list of pathspec(s)\n        (run_id/step_name/task_id) that should be filtered.\n        Note: When `pathspecs` is specified, we expect strict consistency and\n        not eventual consistency in contrast to other modes.\n\n        Parameters\n        ----------\n        run_id : str, optional\n            Run ID to get the tasks from. If not specified, use pathspecs,\n            by default None\n        steps : List[str] , optional\n            Steps to get the tasks from. If run_id is specified, this\n            must also be specified, by default None\n        pathspecs : List[str], optional\n            Full task specs (run_id/step_name/task_id). Can be used instead of\n            specifying run_id and steps, by default None\n        allow_not_done : bool, optional\n            If True, returns the latest attempt of a task even if that attempt\n            wasn't marked as done, by default False\n\n        Returns\n        -------\n        List[TaskDataStore]\n            Task datastores for all the tasks specified.\n        "
        task_urls = []
        if pathspecs:
            task_urls = [self._storage_impl.path_join(self.flow_name, pathspec) for pathspec in pathspecs]
        else:
            run_prefix = self._storage_impl.path_join(self.flow_name, run_id)
            if steps:
                step_urls = [self._storage_impl.path_join(run_prefix, step) for step in steps]
            else:
                step_urls = [step.path for step in self._storage_impl.list_content([run_prefix]) if step.is_file is False]
            task_urls = [task.path for task in self._storage_impl.list_content(step_urls) if task.is_file is False]
        urls = []
        for task_url in task_urls:
            for attempt in range(metaflow_config.MAX_ATTEMPTS):
                for suffix in [TaskDataStore.METADATA_DATA_SUFFIX, TaskDataStore.METADATA_ATTEMPT_SUFFIX, TaskDataStore.METADATA_DONE_SUFFIX]:
                    urls.append(self._storage_impl.path_join(task_url, TaskDataStore.metadata_name_for_attempt(suffix, attempt)))
        latest_started_attempts = {}
        done_attempts = set()
        data_objs = {}
        with self._storage_impl.load_bytes(urls) as get_results:
            for (key, path, meta) in get_results:
                if path is not None:
                    (_, run, step, task, fname) = self._storage_impl.path_split(key)
                    (attempt, fname) = TaskDataStore.parse_attempt_metadata(fname)
                    attempt = int(attempt)
                    if fname == TaskDataStore.METADATA_DONE_SUFFIX:
                        done_attempts.add((run, step, task, attempt))
                    elif fname == TaskDataStore.METADATA_ATTEMPT_SUFFIX:
                        latest_started_attempts[run, step, task] = max(latest_started_attempts.get((run, step, task), 0), attempt)
                    elif fname == TaskDataStore.METADATA_DATA_SUFFIX:
                        with open(path, encoding='utf-8') as f:
                            data_objs[run, step, task, attempt] = json.load(f)
        latest_started_attempts = set(((run, step, task, attempt) for ((run, step, task), attempt) in latest_started_attempts.items()))
        if allow_not_done:
            latest_to_fetch = latest_started_attempts
        else:
            latest_to_fetch = latest_started_attempts & done_attempts
        latest_to_fetch = [(v[0], v[1], v[2], v[3], data_objs.get(v), 'r', allow_not_done) for v in latest_to_fetch]
        return list(itertools.starmap(self.get_task_datastore, latest_to_fetch))

    def get_task_datastore(self, run_id, step_name, task_id, attempt=None, data_metadata=None, mode='r', allow_not_done=False):
        if False:
            i = 10
            return i + 15
        return TaskDataStore(self, run_id, step_name, task_id, attempt=attempt, data_metadata=data_metadata, mode=mode, allow_not_done=allow_not_done)

    def save_data(self, data_iter, len_hint=0):
        if False:
            for i in range(10):
                print('nop')
        'Saves data to the underlying content-addressed store\n\n        Parameters\n        ----------\n        data_iter : Iterator[bytes]\n            Iterator over blobs to save; each item in the list will be saved individually.\n        len_hint : int\n            Estimate of the number of items that will be produced by the iterator,\n            by default 0.\n\n        Returns\n        -------\n        (str, str)\n            Tuple containing the URI to access the saved resource as well as\n            the key needed to retrieve it using load_data. This is returned in\n            the same order as the input.\n        '
        save_results = self.ca_store.save_blobs(data_iter, raw=True, len_hint=len_hint)
        return [(r.uri, r.key) for r in save_results]

    def load_data(self, keys, force_raw=False):
        if False:
            i = 10
            return i + 15
        'Retrieves data from the underlying content-addressed store\n\n        Parameters\n        ----------\n        keys : List[str]\n            Keys to retrieve\n        force_raw : bool, optional\n            Backward compatible mode. Raw data will be properly identified with\n            metadata information but older datastores did not do this. If you\n            know the data should be handled as raw data, set this to True,\n            by default False\n\n        Returns\n        -------\n        Iterator[bytes]\n            Iterator over (key, blob) tuples\n        '
        for (key, blob) in self.ca_store.load_blobs(keys, force_raw=force_raw):
            yield (key, blob)