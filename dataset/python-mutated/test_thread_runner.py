from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from typing import Any
import pytest
from kedro.framework.hooks import _create_hook_manager
from kedro.io import AbstractDataset, DataCatalog, DatasetError, MemoryDataset
from kedro.pipeline import node
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline
from kedro.runner import ThreadRunner
from tests.runner.conftest import exception_fn, identity, return_none, sink, source

class TestValidThreadRunner:

    def test_create_default_data_set(self):
        if False:
            print('Hello World!')
        data_set = ThreadRunner().create_default_data_set('')
        assert isinstance(data_set, MemoryDataset)

    def test_thread_run(self, fan_out_fan_in, catalog):
        if False:
            print('Hello World!')
        catalog.add_feed_dict({'A': 42})
        result = ThreadRunner().run(fan_out_fan_in, catalog)
        assert 'Z' in result
        assert result['Z'] == (42, 42, 42)

    def test_thread_run_with_plugin_manager(self, fan_out_fan_in, catalog):
        if False:
            i = 10
            return i + 15
        catalog.add_feed_dict({'A': 42})
        result = ThreadRunner().run(fan_out_fan_in, catalog, hook_manager=_create_hook_manager())
        assert 'Z' in result
        assert result['Z'] == (42, 42, 42)

    def test_memory_dataset_input(self, fan_out_fan_in):
        if False:
            return 10
        catalog = DataCatalog({'A': MemoryDataset('42')})
        result = ThreadRunner().run(fan_out_fan_in, catalog)
        assert 'Z' in result
        assert result['Z'] == ('42', '42', '42')

class TestMaxWorkers:

    @pytest.mark.parametrize('user_specified_number, expected_number', [(6, 3), (None, 3)])
    def test_specified_max_workers(self, mocker, fan_out_fan_in, catalog, user_specified_number, expected_number):
        if False:
            while True:
                i = 10
        '\n        We initialize the runner with max_workers=4.\n        `fan_out_fan_in` pipeline needs 3 threads.\n        A pool with 3 workers should be used.\n        '
        executor_cls_mock = mocker.patch('kedro.runner.thread_runner.ThreadPoolExecutor', wraps=ThreadPoolExecutor)
        catalog.add_feed_dict({'A': 42})
        result = ThreadRunner(max_workers=user_specified_number).run(fan_out_fan_in, catalog)
        assert result == {'Z': (42, 42, 42)}
        executor_cls_mock.assert_called_once_with(max_workers=expected_number)

    def test_init_with_negative_process_count(self):
        if False:
            return 10
        pattern = 'max_workers should be positive'
        with pytest.raises(ValueError, match=pattern):
            ThreadRunner(max_workers=-1)

class TestIsAsync:

    def test_thread_run(self, fan_out_fan_in, catalog):
        if False:
            for i in range(10):
                print('nop')
        catalog.add_feed_dict({'A': 42})
        pattern = "'ThreadRunner' doesn't support loading and saving the node inputs and outputs asynchronously with threads. Setting 'is_async' to False."
        with pytest.warns(UserWarning, match=pattern):
            result = ThreadRunner(is_async=True).run(fan_out_fan_in, catalog)
        assert 'Z' in result
        assert result['Z'] == (42, 42, 42)

class TestInvalidThreadRunner:

    def test_task_exception(self, fan_out_fan_in, catalog):
        if False:
            print('Hello World!')
        catalog.add_feed_dict(feed_dict={'A': 42})
        pipeline = modular_pipeline([fan_out_fan_in, node(exception_fn, 'Z', 'X')])
        with pytest.raises(Exception, match='test exception'):
            ThreadRunner().run(pipeline, catalog)

    def test_node_returning_none(self):
        if False:
            i = 10
            return i + 15
        pipeline = modular_pipeline([node(identity, 'A', 'B'), node(return_none, 'B', 'C')])
        catalog = DataCatalog({'A': MemoryDataset('42')})
        pattern = "Saving 'None' to a 'Dataset' is not allowed"
        with pytest.raises(DatasetError, match=pattern):
            ThreadRunner().run(pipeline, catalog)

class LoggingDataset(AbstractDataset):

    def __init__(self, log, name, value=None):
        if False:
            while True:
                i = 10
        self.log = log
        self.name = name
        self.value = value

    def _load(self) -> Any:
        if False:
            return 10
        self.log.append(('load', self.name))
        return self.value

    def _save(self, data: Any) -> None:
        if False:
            print('Hello World!')
        self.value = data

    def _release(self) -> None:
        if False:
            i = 10
            return i + 15
        self.log.append(('release', self.name))
        self.value = None

    def _describe(self) -> dict[str, Any]:
        if False:
            while True:
                i = 10
        return {}

class TestThreadRunnerRelease:

    def test_dont_release_inputs_and_outputs(self):
        if False:
            print('Hello World!')
        log = []
        pipeline = modular_pipeline([node(identity, 'in', 'middle'), node(identity, 'middle', 'out')])
        catalog = DataCatalog({'in': LoggingDataset(log, 'in', 'stuff'), 'middle': LoggingDataset(log, 'middle'), 'out': LoggingDataset(log, 'out')})
        ThreadRunner().run(pipeline, catalog)
        assert list(log) == [('load', 'in'), ('load', 'middle'), ('release', 'middle')]

    def test_release_at_earliest_opportunity(self):
        if False:
            return 10
        runner = ThreadRunner()
        log = []
        pipeline = modular_pipeline([node(source, None, 'first'), node(identity, 'first', 'second'), node(sink, 'second', None)])
        catalog = DataCatalog({'first': LoggingDataset(log, 'first'), 'second': LoggingDataset(log, 'second')})
        runner.run(pipeline, catalog)
        assert list(log) == [('load', 'first'), ('release', 'first'), ('load', 'second'), ('release', 'second')]

    def test_count_multiple_loads(self):
        if False:
            for i in range(10):
                print('nop')
        runner = ThreadRunner()
        log = []
        pipeline = modular_pipeline([node(source, None, 'dataset'), node(sink, 'dataset', None, name='bob'), node(sink, 'dataset', None, name='fred')])
        catalog = DataCatalog({'dataset': LoggingDataset(log, 'dataset')})
        runner.run(pipeline, catalog)
        assert list(log) == [('load', 'dataset'), ('load', 'dataset'), ('release', 'dataset')]

    def test_release_transcoded(self):
        if False:
            while True:
                i = 10
        log = []
        pipeline = modular_pipeline([node(source, None, 'ds@save'), node(sink, 'ds@load', None)])
        catalog = DataCatalog({'ds@save': LoggingDataset(log, 'save'), 'ds@load': LoggingDataset(log, 'load')})
        ThreadRunner().run(pipeline, catalog)
        assert list(log) == [('release', 'save'), ('load', 'load'), ('release', 'load')]