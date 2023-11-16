"""``ParallelRunner`` is an ``AbstractRunner`` implementation. It can
be used to run the ``Pipeline`` in parallel groups formed by toposort.
"""
from __future__ import annotations
import multiprocessing
import os
import pickle
import sys
import warnings
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from itertools import chain
from multiprocessing.managers import BaseProxy, SyncManager
from multiprocessing.reduction import ForkingPickler
from pickle import PicklingError
from typing import Any, Iterable
from pluggy import PluginManager
from kedro import KedroDeprecationWarning
from kedro.framework.hooks.manager import _create_hook_manager, _register_hooks, _register_hooks_entry_points
from kedro.framework.project import settings
from kedro.io import DataCatalog, DatasetError, MemoryDataset
from kedro.pipeline import Pipeline
from kedro.pipeline.node import Node
from kedro.runner.runner import AbstractRunner, run_node
_MAX_WINDOWS_WORKERS = 61
_SharedMemoryDataSet: type[_SharedMemoryDataset]

class _SharedMemoryDataset:
    """``_SharedMemoryDataset`` is a wrapper class for a shared MemoryDataset in SyncManager.
    It is not inherited from AbstractDataset class.
    """

    def __init__(self, manager: SyncManager):
        if False:
            print('Hello World!')
        'Creates a new instance of ``_SharedMemoryDataset``,\n        and creates shared memorydataset attribute.\n\n        Args:\n            manager: An instance of multiprocessing manager for shared objects.\n\n        '
        self.shared_memory_dataset = manager.MemoryDataset()

    def __getattr__(self, name):
        if False:
            i = 10
            return i + 15
        if name == '__setstate__':
            raise AttributeError()
        return getattr(self.shared_memory_dataset, name)

    def save(self, data: Any):
        if False:
            for i in range(10):
                print('nop')
        'Calls save method of a shared MemoryDataset in SyncManager.'
        try:
            self.shared_memory_dataset.save(data)
        except Exception as exc:
            try:
                pickle.dumps(data)
            except Exception as serialisation_exc:
                raise DatasetError(f'{str(data.__class__)} cannot be serialised. ParallelRunner implicit memory datasets can only be used with serialisable data') from serialisation_exc
            raise exc

def __getattr__(name):
    if False:
        i = 10
        return i + 15
    if name == '_SharedMemoryDataSet':
        alias = _SharedMemoryDataset
        warnings.warn(f'{repr(name)} has been renamed to {repr(alias.__name__)}, and the alias will be removed in Kedro 0.19.0', KedroDeprecationWarning, stacklevel=2)
        return alias
    raise AttributeError(f'module {repr(__name__)} has no attribute {repr(name)}')

class ParallelRunnerManager(SyncManager):
    """``ParallelRunnerManager`` is used to create shared ``MemoryDataset``
    objects as default data sets in a pipeline.
    """
ParallelRunnerManager.register('MemoryDataset', MemoryDataset)

def _bootstrap_subprocess(package_name: str, logging_config: dict[str, Any]):
    if False:
        while True:
            i = 10
    from kedro.framework.project import configure_logging, configure_project
    configure_project(package_name)
    configure_logging(logging_config)

def _run_node_synchronization(node: Node, catalog: DataCatalog, is_async: bool=False, session_id: str=None, package_name: str=None, logging_config: dict[str, Any]=None) -> Node:
    if False:
        for i in range(10):
            print('nop')
    "Run a single `Node` with inputs from and outputs to the `catalog`.\n\n    A ``PluginManager`` instance is created in each subprocess because the\n    ``PluginManager`` can't be serialised.\n\n    Args:\n        node: The ``Node`` to run.\n        catalog: A ``DataCatalog`` containing the node's inputs and outputs.\n        is_async: If True, the node inputs and outputs are loaded and saved\n            asynchronously with threads. Defaults to False.\n        session_id: The session id of the pipeline run.\n        package_name: The name of the project Python package.\n        logging_config: A dictionary containing logging configuration.\n\n    Returns:\n        The node argument.\n\n    "
    if multiprocessing.get_start_method() == 'spawn' and package_name:
        _bootstrap_subprocess(package_name, logging_config)
    hook_manager = _create_hook_manager()
    _register_hooks(hook_manager, settings.HOOKS)
    _register_hooks_entry_points(hook_manager, settings.DISABLE_HOOKS_FOR_PLUGINS)
    return run_node(node, catalog, hook_manager, is_async, session_id)

class ParallelRunner(AbstractRunner):
    """``ParallelRunner`` is an ``AbstractRunner`` implementation. It can
    be used to run the ``Pipeline`` in parallel groups formed by toposort.
    Please note that this `runner` implementation validates dataset using the
    ``_validate_catalog`` method, which checks if any of the datasets are
    single process only using the `_SINGLE_PROCESS` dataset attribute.
    """

    def __init__(self, max_workers: int=None, is_async: bool=False):
        if False:
            print('Hello World!')
        '\n        Instantiates the runner by creating a Manager.\n\n        Args:\n            max_workers: Number of worker processes to spawn. If not set,\n                calculated automatically based on the pipeline configuration\n                and CPU core count. On windows machines, the max_workers value\n                cannot be larger than 61 and will be set to min(61, max_workers).\n            is_async: If True, the node inputs and outputs are loaded and saved\n                asynchronously with threads. Defaults to False.\n\n        Raises:\n            ValueError: bad parameters passed\n        '
        super().__init__(is_async=is_async)
        self._manager = ParallelRunnerManager()
        self._manager.start()
        if max_workers is None:
            max_workers = os.cpu_count() or 1
            if sys.platform == 'win32':
                max_workers = min(_MAX_WINDOWS_WORKERS, max_workers)
        self._max_workers = max_workers

    def __del__(self):
        if False:
            return 10
        self._manager.shutdown()

    def create_default_data_set(self, ds_name: str) -> _SharedMemoryDataset:
        if False:
            return 10
        'Factory method for creating the default dataset for the runner.\n\n        Args:\n            ds_name: Name of the missing dataset.\n\n        Returns:\n            An instance of ``_SharedMemoryDataset`` to be used for all\n            unregistered datasets.\n\n        '
        return _SharedMemoryDataset(self._manager)

    @classmethod
    def _validate_nodes(cls, nodes: Iterable[Node]):
        if False:
            return 10
        'Ensure all tasks are serialisable.'
        unserialisable = []
        for node in nodes:
            try:
                ForkingPickler.dumps(node)
            except (AttributeError, PicklingError):
                unserialisable.append(node)
        if unserialisable:
            raise AttributeError(f'The following nodes cannot be serialised: {sorted(unserialisable)}\nIn order to utilize multiprocessing you need to make sure all nodes are serialisable, i.e. nodes should not include lambda functions, nested functions, closures, etc.\nIf you are using custom decorators ensure they are correctly decorated using functools.wraps().')

    @classmethod
    def _validate_catalog(cls, catalog: DataCatalog, pipeline: Pipeline):
        if False:
            return 10
        'Ensure that all data sets are serialisable and that we do not have\n        any non proxied memory data sets being used as outputs as their content\n        will not be synchronized across threads.\n        '
        data_sets = catalog._data_sets
        unserialisable = []
        for (name, data_set) in data_sets.items():
            if getattr(data_set, '_SINGLE_PROCESS', False):
                unserialisable.append(name)
                continue
            try:
                ForkingPickler.dumps(data_set)
            except (AttributeError, PicklingError):
                unserialisable.append(name)
        if unserialisable:
            raise AttributeError(f'The following data sets cannot be used with multiprocessing: {sorted(unserialisable)}\nIn order to utilize multiprocessing you need to make sure all data sets are serialisable, i.e. data sets should not make use of lambda functions, nested functions, closures etc.\nIf you are using custom decorators ensure they are correctly decorated using functools.wraps().')
        memory_datasets = []
        for (name, data_set) in data_sets.items():
            if name in pipeline.all_outputs() and isinstance(data_set, MemoryDataset) and (not isinstance(data_set, BaseProxy)):
                memory_datasets.append(name)
        if memory_datasets:
            raise AttributeError(f'The following data sets are memory data sets: {sorted(memory_datasets)}\nParallelRunner does not support output to externally created MemoryDatasets')

    def _get_required_workers_count(self, pipeline: Pipeline):
        if False:
            i = 10
            return i + 15
        '\n        Calculate the max number of processes required for the pipeline,\n        limit to the number of CPU cores.\n        '
        required_processes = len(pipeline.nodes) - len(pipeline.grouped_nodes) + 1
        return min(required_processes, self._max_workers)

    def _run(self, pipeline: Pipeline, catalog: DataCatalog, hook_manager: PluginManager, session_id: str=None) -> None:
        if False:
            while True:
                i = 10
        'The abstract interface for running pipelines.\n\n        Args:\n            pipeline: The ``Pipeline`` to run.\n            catalog: The ``DataCatalog`` from which to fetch data.\n            hook_manager: The ``PluginManager`` to activate hooks.\n            session_id: The id of the session.\n\n        Raises:\n            AttributeError: When the provided pipeline is not suitable for\n                parallel execution.\n            RuntimeError: If the runner is unable to schedule the execution of\n                all pipeline nodes.\n            Exception: In case of any downstream node failure.\n\n        '
        nodes = pipeline.nodes
        self._validate_catalog(catalog, pipeline)
        self._validate_nodes(nodes)
        load_counts = Counter(chain.from_iterable((n.inputs for n in nodes)))
        node_dependencies = pipeline.node_dependencies
        todo_nodes = set(node_dependencies.keys())
        done_nodes: set[Node] = set()
        futures = set()
        done = None
        max_workers = self._get_required_workers_count(pipeline)
        from kedro.framework.project import LOGGING, PACKAGE_NAME
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            while True:
                ready = {n for n in todo_nodes if node_dependencies[n] <= done_nodes}
                todo_nodes -= ready
                for node in ready:
                    futures.add(pool.submit(_run_node_synchronization, node, catalog, self._is_async, session_id, package_name=PACKAGE_NAME, logging_config=LOGGING))
                if not futures:
                    if todo_nodes:
                        debug_data = {'todo_nodes': todo_nodes, 'done_nodes': done_nodes, 'ready_nodes': ready, 'done_futures': done}
                        debug_data_str = '\n'.join((f'{k} = {v}' for (k, v) in debug_data.items()))
                        raise RuntimeError(f'Unable to schedule new tasks although some nodes have not been run:\n{debug_data_str}')
                    break
                (done, futures) = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    node = future.result()
                    done_nodes.add(node)
                    for data_set in node.inputs:
                        load_counts[data_set] -= 1
                        if load_counts[data_set] < 1 and data_set not in pipeline.inputs():
                            catalog.release(data_set)
                    for data_set in node.outputs:
                        if load_counts[data_set] < 1 and data_set not in pipeline.outputs():
                            catalog.release(data_set)