"""``SequentialRunner`` is an ``AbstractRunner`` implementation. It can be
used to run the ``Pipeline`` in a sequential manner using a topological sort
of provided nodes.
"""
from collections import Counter
from itertools import chain
from pluggy import PluginManager
from kedro.io import AbstractDataset, DataCatalog, MemoryDataset
from kedro.pipeline import Pipeline
from kedro.runner.runner import AbstractRunner, run_node

class SequentialRunner(AbstractRunner):
    """``SequentialRunner`` is an ``AbstractRunner`` implementation. It can
    be used to run the ``Pipeline`` in a sequential manner using a
    topological sort of provided nodes.
    """

    def __init__(self, is_async: bool=False):
        if False:
            while True:
                i = 10
        'Instantiates the runner classs.\n\n        Args:\n            is_async: If True, the node inputs and outputs are loaded and saved\n                asynchronously with threads. Defaults to False.\n\n        '
        super().__init__(is_async=is_async)

    def create_default_data_set(self, ds_name: str) -> AbstractDataset:
        if False:
            return 10
        'Factory method for creating the default data set for the runner.\n\n        Args:\n            ds_name: Name of the missing data set\n\n        Returns:\n            An instance of an implementation of AbstractDataset to be used\n            for all unregistered data sets.\n\n        '
        return MemoryDataset()

    def _run(self, pipeline: Pipeline, catalog: DataCatalog, hook_manager: PluginManager, session_id: str=None) -> None:
        if False:
            return 10
        'The method implementing sequential pipeline running.\n\n        Args:\n            pipeline: The ``Pipeline`` to run.\n            catalog: The ``DataCatalog`` from which to fetch data.\n            hook_manager: The ``PluginManager`` to activate hooks.\n            session_id: The id of the session.\n\n        Raises:\n            Exception: in case of any downstream node failure.\n        '
        nodes = pipeline.nodes
        done_nodes = set()
        load_counts = Counter(chain.from_iterable((n.inputs for n in nodes)))
        for (exec_index, node) in enumerate(nodes):
            try:
                run_node(node, catalog, hook_manager, self._is_async, session_id)
                done_nodes.add(node)
            except Exception:
                self._suggest_resume_scenario(pipeline, done_nodes, catalog)
                raise
            for data_set in node.inputs:
                load_counts[data_set] -= 1
                if load_counts[data_set] < 1 and data_set not in pipeline.inputs():
                    catalog.release(data_set)
            for data_set in node.outputs:
                if load_counts[data_set] < 1 and data_set not in pipeline.outputs():
                    catalog.release(data_set)
            self._logger.info('Completed %d out of %d tasks', exec_index + 1, len(nodes))