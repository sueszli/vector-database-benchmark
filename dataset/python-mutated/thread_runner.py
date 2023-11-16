"""``ThreadRunner`` is an ``AbstractRunner`` implementation. It can
be used to run the ``Pipeline`` in parallel groups formed by toposort
using threads.
"""
from __future__ import annotations
import warnings
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from itertools import chain
from pluggy import PluginManager
from kedro.io import DataCatalog, MemoryDataset
from kedro.pipeline import Pipeline
from kedro.pipeline.node import Node
from kedro.runner.runner import AbstractRunner, run_node

class ThreadRunner(AbstractRunner):
    """``ThreadRunner`` is an ``AbstractRunner`` implementation. It can
    be used to run the ``Pipeline`` in parallel groups formed by toposort
    using threads.
    """

    def __init__(self, max_workers: int=None, is_async: bool=False):
        if False:
            print('Hello World!')
        "\n        Instantiates the runner.\n\n        Args:\n            max_workers: Number of worker processes to spawn. If not set,\n                calculated automatically based on the pipeline configuration\n                and CPU core count.\n            is_async: If True, set to False, because `ThreadRunner`\n                doesn't support loading and saving the node inputs and\n                outputs asynchronously with threads. Defaults to False.\n\n        Raises:\n            ValueError: bad parameters passed\n        "
        if is_async:
            warnings.warn("'ThreadRunner' doesn't support loading and saving the node inputs and outputs asynchronously with threads. Setting 'is_async' to False.")
        super().__init__(is_async=False)
        if max_workers is not None and max_workers <= 0:
            raise ValueError('max_workers should be positive')
        self._max_workers = max_workers

    def create_default_data_set(self, ds_name: str) -> MemoryDataset:
        if False:
            while True:
                i = 10
        'Factory method for creating the default dataset for the runner.\n\n        Args:\n            ds_name: Name of the missing dataset.\n\n        Returns:\n            An instance of ``MemoryDataset`` to be used for all\n            unregistered datasets.\n\n        '
        return MemoryDataset()

    def _get_required_workers_count(self, pipeline: Pipeline):
        if False:
            i = 10
            return i + 15
        '\n        Calculate the max number of processes required for the pipeline\n        '
        required_threads = len(pipeline.nodes) - len(pipeline.grouped_nodes) + 1
        return min(required_threads, self._max_workers) if self._max_workers else required_threads

    def _run(self, pipeline: Pipeline, catalog: DataCatalog, hook_manager: PluginManager, session_id: str=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        'The abstract interface for running pipelines.\n\n        Args:\n            pipeline: The ``Pipeline`` to run.\n            catalog: The ``DataCatalog`` from which to fetch data.\n            hook_manager: The ``PluginManager`` to activate hooks.\n            session_id: The id of the session.\n\n        Raises:\n            Exception: in case of any downstream node failure.\n\n        '
        nodes = pipeline.nodes
        load_counts = Counter(chain.from_iterable((n.inputs for n in nodes)))
        node_dependencies = pipeline.node_dependencies
        todo_nodes = set(node_dependencies.keys())
        done_nodes: set[Node] = set()
        futures = set()
        done = None
        max_workers = self._get_required_workers_count(pipeline)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            while True:
                ready = {n for n in todo_nodes if node_dependencies[n] <= done_nodes}
                todo_nodes -= ready
                for node in ready:
                    futures.add(pool.submit(run_node, node, catalog, hook_manager, self._is_async, session_id))
                if not futures:
                    assert not todo_nodes, (todo_nodes, done_nodes, ready, done)
                    break
                (done, futures) = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    try:
                        node = future.result()
                    except Exception:
                        self._suggest_resume_scenario(pipeline, done_nodes, catalog)
                        raise
                    done_nodes.add(node)
                    self._logger.info('Completed node: %s', node.name)
                    self._logger.info('Completed %d out of %d tasks', len(done_nodes), len(nodes))
                    for data_set in node.inputs:
                        load_counts[data_set] -= 1
                        if load_counts[data_set] < 1 and data_set not in pipeline.inputs():
                            catalog.release(data_set)
                    for data_set in node.outputs:
                        if load_counts[data_set] < 1 and data_set not in pipeline.outputs():
                            catalog.release(data_set)