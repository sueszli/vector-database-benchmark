from __future__ import annotations
import importlib
import sys
from concurrent.futures.process import ProcessPoolExecutor
from typing import Any
import pytest
from kedro import KedroDeprecationWarning
from kedro.framework.hooks import _create_hook_manager
from kedro.io import AbstractDataset, DataCatalog, DatasetError, LambdaDataset, MemoryDataset
from kedro.pipeline import node
from kedro.pipeline.modular_pipeline import pipeline as modular_pipeline
from kedro.runner import ParallelRunner
from kedro.runner.parallel_runner import _MAX_WINDOWS_WORKERS, ParallelRunnerManager, _run_node_synchronization, _SharedMemoryDataset
from tests.runner.conftest import exception_fn, identity, return_none, return_not_serialisable, sink, source

def test_deprecation():
    if False:
        while True:
            i = 10
    class_name = '_SharedMemoryDataSet'
    with pytest.warns(KedroDeprecationWarning, match=f'{repr(class_name)} has been renamed'):
        getattr(importlib.import_module('kedro.runner.parallel_runner'), class_name)

@pytest.mark.skipif(sys.platform.startswith('win'), reason='Due to bug in parallel runner')
class TestValidParallelRunner:

    def test_create_default_data_set(self):
        if False:
            while True:
                i = 10
        data_set = ParallelRunner().create_default_data_set('')
        assert isinstance(data_set, _SharedMemoryDataset)

    @pytest.mark.parametrize('is_async', [False, True])
    def test_parallel_run(self, is_async, fan_out_fan_in, catalog):
        if False:
            print('Hello World!')
        catalog.add_feed_dict({'A': 42})
        result = ParallelRunner(is_async=is_async).run(fan_out_fan_in, catalog)
        assert 'Z' in result
        assert len(result['Z']) == 3
        assert result['Z'] == (42, 42, 42)

    @pytest.mark.parametrize('is_async', [False, True])
    def test_parallel_run_with_plugin_manager(self, is_async, fan_out_fan_in, catalog):
        if False:
            i = 10
            return i + 15
        catalog.add_feed_dict({'A': 42})
        result = ParallelRunner(is_async=is_async).run(fan_out_fan_in, catalog, hook_manager=_create_hook_manager())
        assert 'Z' in result
        assert len(result['Z']) == 3
        assert result['Z'] == (42, 42, 42)

    @pytest.mark.parametrize('is_async', [False, True])
    def test_memory_dataset_input(self, is_async, fan_out_fan_in):
        if False:
            print('Hello World!')
        pipeline = modular_pipeline([fan_out_fan_in])
        catalog = DataCatalog({'A': MemoryDataset('42')})
        result = ParallelRunner(is_async=is_async).run(pipeline, catalog)
        assert 'Z' in result
        assert len(result['Z']) == 3
        assert result['Z'] == ('42', '42', '42')

@pytest.mark.skipif(sys.platform.startswith('win'), reason='Due to bug in parallel runner')
class TestMaxWorkers:

    @pytest.mark.parametrize('is_async', [False, True])
    @pytest.mark.parametrize('cpu_cores, user_specified_number, expected_number', [(4, 6, 3), (4, None, 3), (2, None, 2), (1, 2, 2)])
    def test_specified_max_workers_bellow_cpu_cores_count(self, is_async, mocker, fan_out_fan_in, catalog, cpu_cores, user_specified_number, expected_number):
        if False:
            while True:
                i = 10
        '\n        The system has 2 cores, but we initialize the runner with max_workers=4.\n        `fan_out_fan_in` pipeline needs 3 processes.\n        A pool with 3 workers should be used.\n        '
        mocker.patch('os.cpu_count', return_value=cpu_cores)
        executor_cls_mock = mocker.patch('kedro.runner.parallel_runner.ProcessPoolExecutor', wraps=ProcessPoolExecutor)
        catalog.add_feed_dict({'A': 42})
        result = ParallelRunner(max_workers=user_specified_number, is_async=is_async).run(fan_out_fan_in, catalog)
        assert result == {'Z': (42, 42, 42)}
        executor_cls_mock.assert_called_once_with(max_workers=expected_number)

    def test_max_worker_windows(self, mocker):
        if False:
            print('Hello World!')
        'The ProcessPoolExecutor on Python 3.7+\n        has a quirk with the max worker number on Windows\n        and requires it to be <=61\n        '
        mocker.patch('os.cpu_count', return_value=100)
        mocker.patch('sys.platform', 'win32')
        parallel_runner = ParallelRunner()
        assert parallel_runner._max_workers == _MAX_WINDOWS_WORKERS

@pytest.mark.skipif(sys.platform.startswith('win'), reason='Due to bug in parallel runner')
@pytest.mark.parametrize('is_async', [False, True])
class TestInvalidParallelRunner:

    def test_task_validation(self, is_async, fan_out_fan_in, catalog):
        if False:
            i = 10
            return i + 15
        'ParallelRunner cannot serialise the lambda function.'
        catalog.add_feed_dict({'A': 42})
        pipeline = modular_pipeline([fan_out_fan_in, node(lambda x: x, 'Z', 'X')])
        with pytest.raises(AttributeError):
            ParallelRunner(is_async=is_async).run(pipeline, catalog)

    def test_task_exception(self, is_async, fan_out_fan_in, catalog):
        if False:
            return 10
        catalog.add_feed_dict(feed_dict={'A': 42})
        pipeline = modular_pipeline([fan_out_fan_in, node(exception_fn, 'Z', 'X')])
        with pytest.raises(Exception, match='test exception'):
            ParallelRunner(is_async=is_async).run(pipeline, catalog)

    def test_memory_dataset_output(self, is_async, fan_out_fan_in):
        if False:
            for i in range(10):
                print('nop')
        'ParallelRunner does not support output to externally\n        created MemoryDatasets.\n        '
        pipeline = modular_pipeline([fan_out_fan_in])
        catalog = DataCatalog({'C': MemoryDataset()}, {'A': 42})
        with pytest.raises(AttributeError, match="['C']"):
            ParallelRunner(is_async=is_async).run(pipeline, catalog)

    def test_node_returning_none(self, is_async):
        if False:
            for i in range(10):
                print('nop')
        pipeline = modular_pipeline([node(identity, 'A', 'B'), node(return_none, 'B', 'C')])
        catalog = DataCatalog({'A': MemoryDataset('42')})
        pattern = "Saving 'None' to a 'Dataset' is not allowed"
        with pytest.raises(DatasetError, match=pattern):
            ParallelRunner(is_async=is_async).run(pipeline, catalog)

    def test_data_set_not_serialisable(self, is_async, fan_out_fan_in):
        if False:
            for i in range(10):
                print('nop')
        'Data set A cannot be serialisable because _load and _save are not\n        defined in global scope.\n        '

        def _load():
            if False:
                while True:
                    i = 10
            return 0

        def _save(arg):
            if False:
                print('Hello World!')
            assert arg == 0
        catalog = DataCatalog({'A': LambdaDataset(load=_load, save=_save)})
        pipeline = modular_pipeline([fan_out_fan_in])
        with pytest.raises(AttributeError, match="['A']"):
            ParallelRunner(is_async=is_async).run(pipeline, catalog)

    def test_memory_dataset_not_serialisable(self, is_async, catalog):
        if False:
            for i in range(10):
                print('nop')
        'Memory dataset cannot be serialisable because of data it stores.'
        data = return_not_serialisable(None)
        pipeline = modular_pipeline([node(return_not_serialisable, 'A', 'B')])
        catalog.add_feed_dict(feed_dict={'A': 42})
        pattern = f'{str(data.__class__)} cannot be serialised. ParallelRunner implicit memory datasets can only be used with serialisable data'
        with pytest.raises(DatasetError, match=pattern):
            ParallelRunner(is_async=is_async).run(pipeline, catalog)

    def test_unable_to_schedule_all_nodes(self, mocker, is_async, fan_out_fan_in, catalog):
        if False:
            print('Hello World!')
        'Test the error raised when `futures` variable is empty,\n        but `todo_nodes` is not (can barely happen in real life).\n        '
        catalog.add_feed_dict({'A': 42})
        runner = ParallelRunner(is_async=is_async)
        real_node_deps = fan_out_fan_in.node_dependencies
        fake_node_deps = {k: {'you_shall_not_pass'} for k in real_node_deps}
        mocker.patch('kedro.pipeline.Pipeline.node_dependencies', new_callable=mocker.PropertyMock, return_value=fake_node_deps)
        pattern = 'Unable to schedule new tasks although some nodes have not been run'
        with pytest.raises(RuntimeError, match=pattern):
            runner.run(fan_out_fan_in, catalog)

class LoggingDataset(AbstractDataset):

    def __init__(self, log, name, value=None):
        if False:
            print('Hello World!')
        self.log = log
        self.name = name
        self.value = value

    def _load(self) -> Any:
        if False:
            for i in range(10):
                print('nop')
        self.log.append(('load', self.name))
        return self.value

    def _save(self, data: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.value = data

    def _release(self) -> None:
        if False:
            while True:
                i = 10
        self.log.append(('release', self.name))
        self.value = None

    def _describe(self) -> dict[str, Any]:
        if False:
            i = 10
            return i + 15
        return {}
if not sys.platform.startswith('win'):
    ParallelRunnerManager.register('LoggingDataset', LoggingDataset)

@pytest.mark.skipif(sys.platform.startswith('win'), reason='Due to bug in parallel runner')
@pytest.mark.parametrize('is_async', [False, True])
class TestParallelRunnerRelease:

    def test_dont_release_inputs_and_outputs(self, is_async):
        if False:
            return 10
        runner = ParallelRunner(is_async=is_async)
        log = runner._manager.list()
        pipeline = modular_pipeline([node(identity, 'in', 'middle'), node(identity, 'middle', 'out')])
        catalog = DataCatalog({'in': runner._manager.LoggingDataset(log, 'in', 'stuff'), 'middle': runner._manager.LoggingDataset(log, 'middle'), 'out': runner._manager.LoggingDataset(log, 'out')})
        ParallelRunner().run(pipeline, catalog)
        assert list(log) == [('load', 'in'), ('load', 'middle'), ('release', 'middle')]

    def test_release_at_earliest_opportunity(self, is_async):
        if False:
            print('Hello World!')
        runner = ParallelRunner(is_async=is_async)
        log = runner._manager.list()
        pipeline = modular_pipeline([node(source, None, 'first'), node(identity, 'first', 'second'), node(sink, 'second', None)])
        catalog = DataCatalog({'first': runner._manager.LoggingDataset(log, 'first'), 'second': runner._manager.LoggingDataset(log, 'second')})
        runner.run(pipeline, catalog)
        assert list(log) == [('load', 'first'), ('release', 'first'), ('load', 'second'), ('release', 'second')]

    def test_count_multiple_loads(self, is_async):
        if False:
            i = 10
            return i + 15
        runner = ParallelRunner(is_async=is_async)
        log = runner._manager.list()
        pipeline = modular_pipeline([node(source, None, 'dataset'), node(sink, 'dataset', None, name='bob'), node(sink, 'dataset', None, name='fred')])
        catalog = DataCatalog({'dataset': runner._manager.LoggingDataset(log, 'dataset')})
        runner.run(pipeline, catalog)
        assert list(log) == [('load', 'dataset'), ('load', 'dataset'), ('release', 'dataset')]

    def test_release_transcoded(self, is_async):
        if False:
            print('Hello World!')
        runner = ParallelRunner(is_async=is_async)
        log = runner._manager.list()
        pipeline = modular_pipeline([node(source, None, 'ds@save'), node(sink, 'ds@load', None)])
        catalog = DataCatalog({'ds@save': LoggingDataset(log, 'save'), 'ds@load': LoggingDataset(log, 'load')})
        ParallelRunner().run(pipeline, catalog)
        assert list(log) == [('release', 'save'), ('load', 'load'), ('release', 'load')]

@pytest.mark.parametrize('is_async', [False, True])
class TestRunNodeSynchronisationHelper:
    """Test class for _run_node_synchronization helper. It is tested manually
    in isolation since it's called in the subprocess, which ParallelRunner
    patches have no access to.
    """

    @pytest.fixture(autouse=True)
    def mock_logging(self, mocker):
        if False:
            while True:
                i = 10
        return mocker.patch('logging.config.dictConfig')

    @pytest.fixture
    def mock_run_node(self, mocker):
        if False:
            return 10
        return mocker.patch('kedro.runner.parallel_runner.run_node')

    @pytest.fixture
    def mock_configure_project(self, mocker):
        if False:
            i = 10
            return i + 15
        return mocker.patch('kedro.framework.project.configure_project')

    def test_package_name_and_logging_provided(self, mock_logging, mock_run_node, mock_configure_project, is_async, mocker):
        if False:
            print('Hello World!')
        mocker.patch('multiprocessing.get_start_method', return_value='spawn')
        node_ = mocker.sentinel.node
        catalog = mocker.sentinel.catalog
        session_id = 'fake_session_id'
        package_name = mocker.sentinel.package_name
        _run_node_synchronization(node_, catalog, is_async, session_id, package_name=package_name, logging_config={'fake_logging_config': True})
        mock_run_node.assert_called_once()
        mock_logging.assert_called_once_with({'fake_logging_config': True})
        mock_configure_project.assert_called_once_with(package_name)

    def test_package_name_not_provided(self, mock_logging, mock_run_node, is_async, mocker):
        if False:
            return 10
        mocker.patch('multiprocessing.get_start_method', return_value='fork')
        node_ = mocker.sentinel.node
        catalog = mocker.sentinel.catalog
        session_id = 'fake_session_id'
        package_name = mocker.sentinel.package_name
        _run_node_synchronization(node_, catalog, is_async, session_id, package_name=package_name)
        mock_run_node.assert_called_once()
        mock_logging.assert_not_called()