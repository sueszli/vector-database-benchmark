from __future__ import annotations
import inspect
import logging
import os
import pathlib
import sys
import textwrap
import zipfile
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Iterator
from unittest import mock
from unittest.mock import patch
import pytest
import time_machine
from sqlalchemy import func
from sqlalchemy.exc import OperationalError
import airflow.example_dags
from airflow import settings
from airflow.exceptions import SerializationError
from airflow.models.dag import DAG, DagModel
from airflow.models.dagbag import DagBag
from airflow.models.serialized_dag import SerializedDagModel
from airflow.serialization.serialized_objects import SerializedDAG
from airflow.utils.dates import timezone as tz
from airflow.utils.session import create_session
from airflow.www.security_appless import ApplessAirflowSecurityManager
from tests import cluster_policies
from tests.models import TEST_DAGS_FOLDER
from tests.test_utils import db
from tests.test_utils.asserts import assert_queries_count
from tests.test_utils.config import conf_vars
pytestmark = pytest.mark.db_test
example_dags_folder = pathlib.Path(airflow.example_dags.__path__[0])

def db_clean_up():
    if False:
        while True:
            i = 10
    db.clear_db_dags()
    db.clear_db_runs()
    db.clear_db_serialized_dags()
    db.clear_dag_specific_permissions()

class TestDagBag:

    def setup_class(self):
        if False:
            for i in range(10):
                print('nop')
        db_clean_up()

    def teardown_class(self):
        if False:
            return 10
        db_clean_up()

    def test_get_existing_dag(self, tmp_path):
        if False:
            print('Hello World!')
        "\n        Test that we're able to parse some example DAGs and retrieve them\n        "
        dagbag = DagBag(dag_folder=os.fspath(tmp_path), include_examples=True)
        some_expected_dag_ids = ['example_bash_operator', 'example_branch_operator']
        for dag_id in some_expected_dag_ids:
            dag = dagbag.get_dag(dag_id)
            assert dag is not None
            assert dag_id == dag.dag_id
        assert dagbag.size() >= 7

    def test_get_non_existing_dag(self, tmp_path):
        if False:
            for i in range(10):
                print('nop')
        '\n        test that retrieving a non existing dag id returns None without crashing\n        '
        dagbag = DagBag(dag_folder=os.fspath(tmp_path), include_examples=False)
        non_existing_dag_id = 'non_existing_dag_id'
        assert dagbag.get_dag(non_existing_dag_id) is None

    def test_serialized_dag_not_existing_doesnt_raise(self, tmp_path):
        if False:
            return 10
        '\n        test that retrieving a non existing dag id returns None without crashing\n        '
        dagbag = DagBag(dag_folder=os.fspath(tmp_path), include_examples=False, read_dags_from_db=True)
        non_existing_dag_id = 'non_existing_dag_id'
        assert dagbag.get_dag(non_existing_dag_id) is None

    def test_dont_load_example(self, tmp_path):
        if False:
            while True:
                i = 10
        '\n        test that the example are not loaded\n        '
        dagbag = DagBag(dag_folder=os.fspath(tmp_path), include_examples=False)
        assert dagbag.size() == 0

    def test_safe_mode_heuristic_match(self, tmp_path):
        if False:
            i = 10
            return i + 15
        'With safe mode enabled, a file matching the discovery heuristics\n        should be discovered.\n        '
        path = tmp_path / 'testfile.py'
        path.write_text('# airflow\n# DAG')
        with conf_vars({('core', 'dags_folder'): os.fspath(path.parent)}):
            dagbag = DagBag(include_examples=False, safe_mode=True)
        assert len(dagbag.dagbag_stats) == 1
        assert dagbag.dagbag_stats[0].file == f'/{path.name}'

    def test_safe_mode_heuristic_mismatch(self, tmp_path):
        if False:
            while True:
                i = 10
        'With safe mode enabled, a file not matching the discovery heuristics\n        should not be discovered.\n        '
        path = tmp_path / 'testfile.py'
        path.write_text('')
        with conf_vars({('core', 'dags_folder'): os.fspath(path.parent)}):
            dagbag = DagBag(include_examples=False, safe_mode=True)
        assert len(dagbag.dagbag_stats) == 0

    def test_safe_mode_disabled(self, tmp_path):
        if False:
            i = 10
            return i + 15
        'With safe mode disabled, an empty python file should be discovered.'
        path = tmp_path / 'testfile.py'
        path.write_text('')
        with conf_vars({('core', 'dags_folder'): os.fspath(path.parent)}):
            dagbag = DagBag(include_examples=False, safe_mode=False)
        assert len(dagbag.dagbag_stats) == 1
        assert dagbag.dagbag_stats[0].file == f'/{path.name}'

    def test_process_file_that_contains_multi_bytes_char(self, tmp_path):
        if False:
            while True:
                i = 10
        "\n        test that we're able to parse file that contains multi-byte char\n        "
        path = tmp_path / 'testfile'
        path.write_text('あ')
        dagbag = DagBag(dag_folder=os.fspath(path.parent), include_examples=False)
        assert [] == dagbag.process_file(os.fspath(path))

    def test_process_file_duplicated_dag_id(self, tmp_path):
        if False:
            for i in range(10):
                print('nop')
        'Loading a DAG with ID that already existed in a DAG bag should result in an import error.'
        dagbag = DagBag(dag_folder=os.fspath(tmp_path), include_examples=False)

        def create_dag():
            if False:
                print('Hello World!')
            from airflow.decorators import dag

            @dag(default_args={'owner': 'owner1'})
            def my_flow():
                if False:
                    while True:
                        i = 10
                pass
            my_dag = my_flow()
        source_lines = [line[12:] for line in inspect.getsource(create_dag).splitlines(keepends=True)[1:]]
        path1 = tmp_path / 'testfile1'
        path2 = tmp_path / 'testfile2'
        path1.write_text(''.join(source_lines))
        path2.write_text(''.join(source_lines))
        found_1 = dagbag.process_file(os.fspath(path1))
        assert len(found_1) == 1 and found_1[0].dag_id == 'my_flow'
        assert dagbag.import_errors == {}
        dags_in_bag = dagbag.dags
        found_2 = dagbag.process_file(os.fspath(path2))
        assert len(found_2) == 0
        assert dagbag.import_errors[os.fspath(path2)].startswith('AirflowDagDuplicatedIdException: Ignoring DAG')
        assert dagbag.dags == dags_in_bag

    def test_zip_skip_log(self, caplog):
        if False:
            print('Hello World!')
        '\n        test the loading of a DAG from within a zip file that skips another file because\n        it doesn\'t have "airflow" and "DAG"\n        '
        caplog.set_level(logging.INFO)
        test_zip_path = os.path.join(TEST_DAGS_FOLDER, 'test_zip.zip')
        dagbag = DagBag(dag_folder=test_zip_path, include_examples=False)
        assert dagbag.has_logged
        assert f'File {test_zip_path}:file_no_airflow_dag.py assumed to contain no DAGs. Skipping.' in caplog.text

    def test_zip(self, tmp_path):
        if False:
            i = 10
            return i + 15
        '\n        test the loading of a DAG within a zip file that includes dependencies\n        '
        syspath_before = deepcopy(sys.path)
        dagbag = DagBag(dag_folder=os.fspath(tmp_path), include_examples=False)
        dagbag.process_file(os.path.join(TEST_DAGS_FOLDER, 'test_zip.zip'))
        assert dagbag.get_dag('test_zip_dag')
        assert sys.path == syspath_before

    @patch('airflow.models.dagbag.timeout')
    @patch('airflow.models.dagbag.settings.get_dagbag_import_timeout')
    def test_process_dag_file_without_timeout(self, mocked_get_dagbag_import_timeout, mocked_timeout, tmp_path):
        if False:
            return 10
        '\n        Test dag file parsing without timeout\n        '
        mocked_get_dagbag_import_timeout.return_value = 0
        dagbag = DagBag(dag_folder=os.fspath(tmp_path), include_examples=False)
        dagbag.process_file(os.path.join(TEST_DAGS_FOLDER, 'test_default_views.py'))
        mocked_timeout.assert_not_called()
        mocked_get_dagbag_import_timeout.return_value = -1
        dagbag = DagBag(dag_folder=os.fspath(tmp_path), include_examples=False)
        dagbag.process_file(os.path.join(TEST_DAGS_FOLDER, 'test_default_views.py'))
        mocked_timeout.assert_not_called()

    @patch('airflow.models.dagbag.timeout')
    @patch('airflow.models.dagbag.settings.get_dagbag_import_timeout')
    def test_process_dag_file_with_non_default_timeout(self, mocked_get_dagbag_import_timeout, mocked_timeout, tmp_path):
        if False:
            return 10
        '\n        Test customized dag file parsing timeout\n        '
        timeout_value = 100
        mocked_get_dagbag_import_timeout.return_value = timeout_value
        assert timeout_value != settings.conf.getfloat('core', 'DAGBAG_IMPORT_TIMEOUT')
        dagbag = DagBag(dag_folder=os.fspath(tmp_path), include_examples=False)
        dagbag.process_file(os.path.join(TEST_DAGS_FOLDER, 'test_default_views.py'))
        mocked_timeout.assert_called_once_with(timeout_value, error_message=mock.ANY)

    @patch('airflow.models.dagbag.settings.get_dagbag_import_timeout')
    def test_check_value_type_from_get_dagbag_import_timeout(self, mocked_get_dagbag_import_timeout, tmp_path):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test correctness of value from get_dagbag_import_timeout\n        '
        mocked_get_dagbag_import_timeout.return_value = '1'
        dagbag = DagBag(dag_folder=os.fspath(tmp_path), include_examples=False)
        with pytest.raises(TypeError, match='Value \\(1\\) from get_dagbag_import_timeout must be int or float'):
            dagbag.process_file(os.path.join(TEST_DAGS_FOLDER, 'test_default_views.py'))

    @pytest.fixture()
    def invalid_cron_dag(self) -> str:
        if False:
            i = 10
            return i + 15
        return os.path.join(TEST_DAGS_FOLDER, 'test_invalid_cron.py')

    @pytest.fixture()
    def invalid_cron_zipped_dag(self, invalid_cron_dag: str, tmp_path: pathlib.Path) -> Iterator[str]:
        if False:
            print('Hello World!')
        zipped = tmp_path / 'test_zip_invalid_cron.zip'
        with zipfile.ZipFile(zipped, 'w') as zf:
            zf.write(invalid_cron_dag, os.path.basename(invalid_cron_dag))
        yield os.fspath(zipped)

    @pytest.mark.parametrize('invalid_dag_name', ['invalid_cron_dag', 'invalid_cron_zipped_dag'])
    def test_process_file_cron_validity_check(self, request: pytest.FixtureRequest, invalid_dag_name: str, tmp_path):
        if False:
            i = 10
            return i + 15
        'test if an invalid cron expression as schedule interval can be identified'
        dagbag = DagBag(dag_folder=os.fspath(tmp_path), include_examples=False)
        assert len(dagbag.import_errors) == 0
        dagbag.process_file(request.getfixturevalue(invalid_dag_name))
        assert len(dagbag.import_errors) == 1
        assert len(dagbag.dags) == 0

    def test_process_file_invalid_param_check(self, tmp_path):
        if False:
            print('Hello World!')
        '\n        test if an invalid param in the dags can be identified\n        '
        invalid_dag_files = ['test_invalid_param.py', 'test_invalid_param2.py', 'test_invalid_param3.py', 'test_invalid_param4.py']
        dagbag = DagBag(dag_folder=os.fspath(tmp_path), include_examples=False)
        assert len(dagbag.import_errors) == 0
        for file in invalid_dag_files:
            dagbag.process_file(os.path.join(TEST_DAGS_FOLDER, file))
        assert len(dagbag.import_errors) == len(invalid_dag_files)
        assert len(dagbag.dags) == 0

    def test_process_file_valid_param_check(self, tmp_path):
        if False:
            print('Hello World!')
        '\n        test if valid params in the dags param can be validated (positive test)\n        '
        valid_dag_files = ['test_valid_param.py', 'test_valid_param2.py']
        dagbag = DagBag(dag_folder=os.fspath(tmp_path), include_examples=False)
        assert len(dagbag.import_errors) == 0
        for file in valid_dag_files:
            dagbag.process_file(os.path.join(TEST_DAGS_FOLDER, file))
        assert len(dagbag.import_errors) == 0
        assert len(dagbag.dags) == len(valid_dag_files)

    @patch.object(DagModel, 'get_current')
    def test_get_dag_without_refresh(self, mock_dagmodel):
        if False:
            while True:
                i = 10
        "\n        Test that, once a DAG is loaded, it doesn't get refreshed again if it\n        hasn't been expired.\n        "
        dag_id = 'example_bash_operator'
        mock_dagmodel.return_value = DagModel()
        mock_dagmodel.return_value.last_expired = None
        mock_dagmodel.return_value.fileloc = 'foo'

        class _TestDagBag(DagBag):
            process_file_calls = 0

            def process_file(self, filepath, only_if_updated=True, safe_mode=True):
                if False:
                    for i in range(10):
                        print('nop')
                if os.path.basename(filepath) == 'example_bash_operator.py':
                    _TestDagBag.process_file_calls += 1
                super().process_file(filepath, only_if_updated, safe_mode)
        dagbag = _TestDagBag(include_examples=True)
        dagbag.process_file_calls
        assert 1 == dagbag.process_file_calls
        assert dagbag.get_dag(dag_id) is not None
        assert 1 == dagbag.process_file_calls

    @pytest.mark.parametrize(('file_to_load', 'expected'), (pytest.param(TEST_DAGS_FOLDER / 'test_zip.zip', {'test_zip_dag': 'dags/test_zip.zip/test_zip.py', 'test_zip_autoregister': 'dags/test_zip.zip/test_zip.py'}, id='test_zip.zip'), pytest.param(pathlib.Path(example_dags_folder) / 'example_bash_operator.py', {'example_bash_operator': 'airflow/example_dags/example_bash_operator.py'}, id='example_bash_operator'), pytest.param(TEST_DAGS_FOLDER / 'test_subdag.py', {'test_subdag_operator': 'dags/test_subdag.py', 'test_subdag_operator.section-1': 'dags/test_subdag.py'}, id='test_subdag_operator')))
    def test_get_dag_registration(self, file_to_load, expected):
        if False:
            for i in range(10):
                print('nop')
        dagbag = DagBag(dag_folder=os.devnull, include_examples=False)
        dagbag.process_file(os.fspath(file_to_load))
        for (dag_id, path) in expected.items():
            dag = dagbag.get_dag(dag_id)
            assert dag, f'{dag_id} was bagged'
            assert dag.fileloc.endswith(path)

    def test_dag_registration_with_failure(self):
        if False:
            print('Hello World!')
        dagbag = DagBag(dag_folder=os.devnull, include_examples=False)
        found = dagbag.process_file(str(TEST_DAGS_FOLDER / 'test_invalid_dup_task.py'))
        assert [] == found

    @pytest.fixture()
    def zip_with_valid_dag_and_dup_tasks(self, tmp_path: pathlib.Path) -> Iterator[str]:
        if False:
            for i in range(10):
                print('nop')
        failing_dag_file = TEST_DAGS_FOLDER / 'test_invalid_dup_task.py'
        working_dag_file = TEST_DAGS_FOLDER / 'test_example_bash_operator.py'
        zipped = tmp_path / 'test_zip_invalid_dup_task.zip'
        with zipfile.ZipFile(zipped, 'w') as zf:
            zf.write(failing_dag_file, failing_dag_file.name)
            zf.write(working_dag_file, working_dag_file.name)
        yield os.fspath(zipped)

    def test_dag_registration_with_failure_zipped(self, zip_with_valid_dag_and_dup_tasks):
        if False:
            print('Hello World!')
        dagbag = DagBag(dag_folder=os.devnull, include_examples=False)
        found = dagbag.process_file(zip_with_valid_dag_and_dup_tasks)
        assert 1 == len(found)
        assert ['test_example_bash_operator'] == [dag.dag_id for dag in found]

    @patch.object(DagModel, 'get_current')
    def test_refresh_py_dag(self, mock_dagmodel, tmp_path):
        if False:
            print('Hello World!')
        '\n        Test that we can refresh an ordinary .py DAG\n        '
        dag_id = 'example_bash_operator'
        fileloc = str(example_dags_folder / 'example_bash_operator.py')
        mock_dagmodel.return_value = DagModel()
        mock_dagmodel.return_value.last_expired = datetime.max.replace(tzinfo=timezone.utc)
        mock_dagmodel.return_value.fileloc = fileloc

        class _TestDagBag(DagBag):
            process_file_calls = 0

            def process_file(self, filepath, only_if_updated=True, safe_mode=True):
                if False:
                    for i in range(10):
                        print('nop')
                if filepath == fileloc:
                    _TestDagBag.process_file_calls += 1
                return super().process_file(filepath, only_if_updated, safe_mode)
        dagbag = _TestDagBag(dag_folder=os.fspath(tmp_path), include_examples=True)
        assert 1 == dagbag.process_file_calls
        dag = dagbag.get_dag(dag_id)
        assert dag is not None
        assert dag_id == dag.dag_id
        assert 2 == dagbag.process_file_calls

    @patch.object(DagModel, 'get_current')
    def test_refresh_packaged_dag(self, mock_dagmodel):
        if False:
            i = 10
            return i + 15
        '\n        Test that we can refresh a packaged DAG\n        '
        dag_id = 'test_zip_dag'
        fileloc = os.path.realpath(os.path.join(TEST_DAGS_FOLDER, 'test_zip.zip/test_zip.py'))
        mock_dagmodel.return_value = DagModel()
        mock_dagmodel.return_value.last_expired = datetime.max.replace(tzinfo=timezone.utc)
        mock_dagmodel.return_value.fileloc = fileloc

        class _TestDagBag(DagBag):
            process_file_calls = 0

            def process_file(self, filepath, only_if_updated=True, safe_mode=True):
                if False:
                    i = 10
                    return i + 15
                if filepath in fileloc:
                    _TestDagBag.process_file_calls += 1
                return super().process_file(filepath, only_if_updated, safe_mode)
        dagbag = _TestDagBag(dag_folder=os.path.realpath(TEST_DAGS_FOLDER), include_examples=False)
        assert 1 == dagbag.process_file_calls
        dag = dagbag.get_dag(dag_id)
        assert dag is not None
        assert dag_id == dag.dag_id
        assert 2 == dagbag.process_file_calls

    def test_dag_removed_if_serialized_dag_is_removed(self, dag_maker, tmp_path):
        if False:
            while True:
                i = 10
        '\n        Test that if a DAG does not exist in serialized_dag table (as the DAG file was removed),\n        remove dags from the DagBag\n        '
        from airflow.operators.empty import EmptyOperator
        with dag_maker(dag_id='test_dag_removed_if_serialized_dag_is_removed', schedule=None, start_date=tz.datetime(2021, 10, 12)) as dag:
            EmptyOperator(task_id='task_1')
        dag_maker.create_dagrun()
        dagbag = DagBag(dag_folder=os.fspath(tmp_path), include_examples=False, read_dags_from_db=True)
        dagbag.dags = {dag.dag_id: SerializedDAG.from_dict(SerializedDAG.to_dict(dag))}
        dagbag.dags_last_fetched = {dag.dag_id: tz.utcnow() - timedelta(minutes=2)}
        dagbag.dags_hash = {dag.dag_id: mock.ANY}
        assert SerializedDagModel.has_dag(dag.dag_id) is False
        assert dagbag.get_dag(dag.dag_id) is None
        assert dag.dag_id not in dagbag.dags
        assert dag.dag_id not in dagbag.dags_last_fetched
        assert dag.dag_id not in dagbag.dags_hash

    def process_dag(self, create_dag, tmp_path):
        if False:
            i = 10
            return i + 15
        '\n        Helper method to process a file generated from the input create_dag function.\n        '
        source = textwrap.dedent(''.join(inspect.getsource(create_dag).splitlines(True)[1:-1]))
        path = tmp_path / 'testfile'
        path.write_text(source)
        dagbag = DagBag(dag_folder=os.fspath(path.parent), include_examples=False)
        found_dags = dagbag.process_file(os.fspath(path))
        return (dagbag, found_dags, os.fspath(path))

    def validate_dags(self, expected_parent_dag, actual_found_dags, actual_dagbag, should_be_found=True):
        if False:
            while True:
                i = 10
        expected_dag_ids = [dag.dag_id for dag in expected_parent_dag.subdags]
        expected_dag_ids.append(expected_parent_dag.dag_id)
        actual_found_dag_ids = [dag.dag_id for dag in actual_found_dags]
        for dag_id in expected_dag_ids:
            actual_dagbag.log.info('validating %s', dag_id)
            assert (dag_id in actual_found_dag_ids) == should_be_found, f'''dag "{dag_id}" should {('' if should_be_found else 'not ')}have been found after processing dag "{expected_parent_dag.dag_id}"'''
            assert (dag_id in actual_dagbag.dags) == should_be_found, f'''dag "{dag_id}" should {('' if should_be_found else 'not ')}be in dagbag.dags after processing dag "{expected_parent_dag.dag_id}"'''

    def test_load_subdags(self, tmp_path):
        if False:
            while True:
                i = 10

        def standard_subdag():
            if False:
                while True:
                    i = 10
            import datetime
            from airflow.models.dag import DAG
            from airflow.operators.empty import EmptyOperator
            from airflow.operators.subdag import SubDagOperator
            dag_name = 'parent'
            default_args = {'owner': 'owner1', 'start_date': datetime.datetime(2016, 1, 1)}
            dag = DAG(dag_name, default_args=default_args)
            with dag:

                def subdag_0():
                    if False:
                        i = 10
                        return i + 15
                    subdag_0 = DAG('parent.op_subdag_0', default_args=default_args)
                    EmptyOperator(task_id='subdag_0.task', dag=subdag_0)
                    return subdag_0

                def subdag_1():
                    if False:
                        return 10
                    subdag_1 = DAG('parent.op_subdag_1', default_args=default_args)
                    EmptyOperator(task_id='subdag_1.task', dag=subdag_1)
                    return subdag_1
                op_subdag_0 = SubDagOperator(task_id='op_subdag_0', dag=dag, subdag=subdag_0())
                op_subdag_1 = SubDagOperator(task_id='op_subdag_1', dag=dag, subdag=subdag_1())
                op_a = EmptyOperator(task_id='A')
                op_a.set_downstream(op_subdag_0)
                op_a.set_downstream(op_subdag_1)
            return dag
        test_dag = standard_subdag()
        assert len(test_dag.subdags) == 2
        (dagbag, found_dags, _) = self.process_dag(standard_subdag, tmp_path)
        self.validate_dags(test_dag, found_dags, dagbag)

        def nested_subdags():
            if False:
                print('Hello World!')
            import datetime
            from airflow.models.dag import DAG
            from airflow.operators.empty import EmptyOperator
            from airflow.operators.subdag import SubDagOperator
            dag_name = 'parent'
            default_args = {'owner': 'owner1', 'start_date': datetime.datetime(2016, 1, 1)}
            dag = DAG(dag_name, default_args=default_args)
            with dag:

                def subdag_a():
                    if False:
                        while True:
                            i = 10
                    subdag_a = DAG('parent.op_subdag_0.opSubdag_A', default_args=default_args)
                    EmptyOperator(task_id='subdag_a.task', dag=subdag_a)
                    return subdag_a

                def subdag_b():
                    if False:
                        for i in range(10):
                            print('nop')
                    subdag_b = DAG('parent.op_subdag_0.opSubdag_B', default_args=default_args)
                    EmptyOperator(task_id='subdag_b.task', dag=subdag_b)
                    return subdag_b

                def subdag_c():
                    if False:
                        while True:
                            i = 10
                    subdag_c = DAG('parent.op_subdag_1.opSubdag_C', default_args=default_args)
                    EmptyOperator(task_id='subdag_c.task', dag=subdag_c)
                    return subdag_c

                def subdag_d():
                    if False:
                        print('Hello World!')
                    subdag_d = DAG('parent.op_subdag_1.opSubdag_D', default_args=default_args)
                    EmptyOperator(task_id='subdag_d.task', dag=subdag_d)
                    return subdag_d

                def subdag_0():
                    if False:
                        i = 10
                        return i + 15
                    subdag_0 = DAG('parent.op_subdag_0', default_args=default_args)
                    SubDagOperator(task_id='opSubdag_A', dag=subdag_0, subdag=subdag_a())
                    SubDagOperator(task_id='opSubdag_B', dag=subdag_0, subdag=subdag_b())
                    return subdag_0

                def subdag_1():
                    if False:
                        while True:
                            i = 10
                    subdag_1 = DAG('parent.op_subdag_1', default_args=default_args)
                    SubDagOperator(task_id='opSubdag_C', dag=subdag_1, subdag=subdag_c())
                    SubDagOperator(task_id='opSubdag_D', dag=subdag_1, subdag=subdag_d())
                    return subdag_1
                op_subdag_0 = SubDagOperator(task_id='op_subdag_0', dag=dag, subdag=subdag_0())
                op_subdag_1 = SubDagOperator(task_id='op_subdag_1', dag=dag, subdag=subdag_1())
                op_a = EmptyOperator(task_id='A')
                op_a.set_downstream(op_subdag_0)
                op_a.set_downstream(op_subdag_1)
            return dag
        test_dag = nested_subdags()
        assert len(test_dag.subdags) == 6
        (dagbag, found_dags, filename) = self.process_dag(nested_subdags, tmp_path)
        self.validate_dags(test_dag, found_dags, dagbag)
        for dag in dagbag.dags.values():
            assert dag.fileloc == filename

    def test_skip_cycle_dags(self, tmp_path):
        if False:
            i = 10
            return i + 15
        "\n        Don't crash when loading an invalid (contains a cycle) DAG file.\n        Don't load the dag into the DagBag either\n        "

        def basic_cycle():
            if False:
                for i in range(10):
                    print('nop')
            import datetime
            from airflow.models.dag import DAG
            from airflow.operators.empty import EmptyOperator
            dag_name = 'cycle_dag'
            default_args = {'owner': 'owner1', 'start_date': datetime.datetime(2016, 1, 1)}
            dag = DAG(dag_name, default_args=default_args)
            with dag:
                op_a = EmptyOperator(task_id='A')
                op_a.set_downstream(op_a)
            return dag
        test_dag = basic_cycle()
        assert len(test_dag.subdags) == 0
        (dagbag, found_dags, file_path) = self.process_dag(basic_cycle, tmp_path)
        self.validate_dags(test_dag, found_dags, dagbag, should_be_found=False)
        assert file_path in dagbag.import_errors

        def nested_subdag_cycle():
            if False:
                print('Hello World!')
            import datetime
            from airflow.models.dag import DAG
            from airflow.operators.empty import EmptyOperator
            from airflow.operators.subdag import SubDagOperator
            dag_name = 'nested_cycle'
            default_args = {'owner': 'owner1', 'start_date': datetime.datetime(2016, 1, 1)}
            dag = DAG(dag_name, default_args=default_args)
            with dag:

                def subdag_a():
                    if False:
                        i = 10
                        return i + 15
                    subdag_a = DAG('nested_cycle.op_subdag_0.opSubdag_A', default_args=default_args)
                    EmptyOperator(task_id='subdag_a.task', dag=subdag_a)
                    return subdag_a

                def subdag_b():
                    if False:
                        return 10
                    subdag_b = DAG('nested_cycle.op_subdag_0.opSubdag_B', default_args=default_args)
                    EmptyOperator(task_id='subdag_b.task', dag=subdag_b)
                    return subdag_b

                def subdag_c():
                    if False:
                        i = 10
                        return i + 15
                    subdag_c = DAG('nested_cycle.op_subdag_1.opSubdag_C', default_args=default_args)
                    op_subdag_c_task = EmptyOperator(task_id='subdag_c.task', dag=subdag_c)
                    op_subdag_c_task.set_downstream(op_subdag_c_task)
                    return subdag_c

                def subdag_d():
                    if False:
                        while True:
                            i = 10
                    subdag_d = DAG('nested_cycle.op_subdag_1.opSubdag_D', default_args=default_args)
                    EmptyOperator(task_id='subdag_d.task', dag=subdag_d)
                    return subdag_d

                def subdag_0():
                    if False:
                        for i in range(10):
                            print('nop')
                    subdag_0 = DAG('nested_cycle.op_subdag_0', default_args=default_args)
                    SubDagOperator(task_id='opSubdag_A', dag=subdag_0, subdag=subdag_a())
                    SubDagOperator(task_id='opSubdag_B', dag=subdag_0, subdag=subdag_b())
                    return subdag_0

                def subdag_1():
                    if False:
                        while True:
                            i = 10
                    subdag_1 = DAG('nested_cycle.op_subdag_1', default_args=default_args)
                    SubDagOperator(task_id='opSubdag_C', dag=subdag_1, subdag=subdag_c())
                    SubDagOperator(task_id='opSubdag_D', dag=subdag_1, subdag=subdag_d())
                    return subdag_1
                op_subdag_0 = SubDagOperator(task_id='op_subdag_0', dag=dag, subdag=subdag_0())
                op_subdag_1 = SubDagOperator(task_id='op_subdag_1', dag=dag, subdag=subdag_1())
                op_a = EmptyOperator(task_id='A')
                op_a.set_downstream(op_subdag_0)
                op_a.set_downstream(op_subdag_1)
            return dag
        test_dag = nested_subdag_cycle()
        assert len(test_dag.subdags) == 6
        (dagbag, found_dags, file_path) = self.process_dag(nested_subdag_cycle, tmp_path)
        self.validate_dags(test_dag, found_dags, dagbag, should_be_found=False)
        assert file_path in dagbag.import_errors

    def test_process_file_with_none(self, tmp_path):
        if False:
            print('Hello World!')
        '\n        test that process_file can handle Nones\n        '
        dagbag = DagBag(dag_folder=os.fspath(tmp_path), include_examples=False)
        assert [] == dagbag.process_file(None)

    def test_deactivate_unknown_dags(self):
        if False:
            return 10
        '\n        Test that dag_ids not passed into deactivate_unknown_dags\n        are deactivated when function is invoked\n        '
        dagbag = DagBag(include_examples=True)
        dag_id = 'test_deactivate_unknown_dags'
        expected_active_dags = dagbag.dags.keys()
        model_before = DagModel(dag_id=dag_id, is_active=True)
        with create_session() as session:
            session.merge(model_before)
        DAG.deactivate_unknown_dags(expected_active_dags)
        after_model = DagModel.get_dagmodel(dag_id)
        assert model_before.is_active
        assert not after_model.is_active
        with create_session() as session:
            session.query(DagModel).filter(DagModel.dag_id == 'test_deactivate_unknown_dags').delete()

    def test_serialized_dags_are_written_to_db_on_sync(self):
        if False:
            while True:
                i = 10
        '\n        Test that when dagbag.sync_to_db is called the DAGs are Serialized and written to DB\n        even when dagbag.read_dags_from_db is False\n        '
        with create_session() as session:
            serialized_dags_count = session.query(func.count(SerializedDagModel.dag_id)).scalar()
            assert serialized_dags_count == 0
            dagbag = DagBag(dag_folder=os.path.join(TEST_DAGS_FOLDER, 'test_example_bash_operator.py'), include_examples=False)
            dagbag.sync_to_db()
            assert not dagbag.read_dags_from_db
            new_serialized_dags_count = session.query(func.count(SerializedDagModel.dag_id)).scalar()
            assert new_serialized_dags_count == 1

    @patch('airflow.models.serialized_dag.SerializedDagModel.write_dag')
    def test_serialized_dag_errors_are_import_errors(self, mock_serialize, caplog):
        if False:
            print('Hello World!')
        '\n        Test that errors serializing a DAG are recorded as import_errors in the DB\n        '
        mock_serialize.side_effect = SerializationError
        with create_session() as session:
            path = os.path.join(TEST_DAGS_FOLDER, 'test_example_bash_operator.py')
            dagbag = DagBag(dag_folder=path, include_examples=False)
            assert dagbag.import_errors == {}
            caplog.set_level(logging.ERROR)
            dagbag.sync_to_db(session=session)
            assert 'SerializationError' in caplog.text
            assert path in dagbag.import_errors
            err = dagbag.import_errors[path]
            assert 'SerializationError' in err
            session.rollback()

    @patch('airflow.models.dagbag.DagBag.collect_dags')
    @patch('airflow.models.serialized_dag.SerializedDagModel.write_dag')
    @patch('airflow.models.dag.DAG.bulk_write_to_db')
    def test_sync_to_db_is_retried(self, mock_bulk_write_to_db, mock_s10n_write_dag, mock_collect_dags):
        if False:
            for i in range(10):
                print('nop')
        'Test that dagbag.sync_to_db is retried on OperationalError'
        dagbag = DagBag('/dev/null')
        mock_dag = mock.MagicMock(spec=DAG)
        mock_dag.is_subdag = False
        dagbag.dags['mock_dag'] = mock_dag
        op_error = OperationalError(statement=mock.ANY, params=mock.ANY, orig=mock.ANY)
        side_effect = [op_error, op_error, mock.ANY]
        mock_bulk_write_to_db.side_effect = side_effect
        mock_session = mock.MagicMock()
        dagbag.sync_to_db(session=mock_session)
        mock_bulk_write_to_db.assert_has_calls([mock.call(mock.ANY, processor_subdir=None, session=mock.ANY), mock.call(mock.ANY, processor_subdir=None, session=mock.ANY), mock.call(mock.ANY, processor_subdir=None, session=mock.ANY)])
        mock_session.rollback.assert_has_calls([mock.call(), mock.call()])
        mock_s10n_write_dag.assert_has_calls([mock.call(mock_dag, min_update_interval=mock.ANY, session=mock_session)])

    @patch('airflow.models.dagbag.settings.MIN_SERIALIZED_DAG_UPDATE_INTERVAL', 5)
    @patch('airflow.models.dagbag.DagBag._sync_perm_for_dag')
    def test_sync_to_db_syncs_dag_specific_perms_on_update(self, mock_sync_perm_for_dag):
        if False:
            return 10
        '\n        Test that dagbag.sync_to_db will sync DAG specific permissions when a DAG is\n        new or updated\n        '
        db_clean_up()
        session = settings.Session()
        with time_machine.travel(tz.datetime(2020, 1, 5, 0, 0, 0), tick=False) as frozen_time:
            dagbag = DagBag(dag_folder=os.path.join(TEST_DAGS_FOLDER, 'test_example_bash_operator.py'), include_examples=False)

            def _sync_to_db():
                if False:
                    i = 10
                    return i + 15
                mock_sync_perm_for_dag.reset_mock()
                frozen_time.shift(20)
                dagbag.sync_to_db(session=session)
            dag = dagbag.dags['test_example_bash_operator']
            _sync_to_db()
            mock_sync_perm_for_dag.assert_called_once_with(dag, session=session)
            _sync_to_db()
            mock_sync_perm_for_dag.assert_not_called()
            dag.tags = ['new_tag']
            _sync_to_db()
            mock_sync_perm_for_dag.assert_called_once_with(dag, session=session)

    @patch('airflow.www.security_appless.ApplessAirflowSecurityManager')
    def test_sync_perm_for_dag(self, mock_security_manager):
        if False:
            return 10
        '\n        Test that dagbag._sync_perm_for_dag will call ApplessAirflowSecurityManager.sync_perm_for_dag\n        '
        db_clean_up()
        with create_session() as session:
            security_manager = ApplessAirflowSecurityManager(session)
            mock_sync_perm_for_dag = mock_security_manager.return_value.sync_perm_for_dag
            mock_sync_perm_for_dag.side_effect = security_manager.sync_perm_for_dag
            dagbag = DagBag(dag_folder=os.path.join(TEST_DAGS_FOLDER, 'test_example_bash_operator.py'), include_examples=False)
            dag = dagbag.dags['test_example_bash_operator']

            def _sync_perms():
                if False:
                    i = 10
                    return i + 15
                mock_sync_perm_for_dag.reset_mock()
                DagBag._sync_perm_for_dag(dag, session=session)
            _sync_perms()
            mock_sync_perm_for_dag.assert_called_once_with('test_example_bash_operator', None)
            _sync_perms()
            mock_sync_perm_for_dag.assert_called_once_with('test_example_bash_operator', None)
            dag.access_control = {'Public': {'can_read'}}
            _sync_perms()
            mock_sync_perm_for_dag.assert_called_once_with('test_example_bash_operator', {'Public': {'can_read'}})

    @patch('airflow.models.dagbag.settings.MIN_SERIALIZED_DAG_UPDATE_INTERVAL', 5)
    @patch('airflow.models.dagbag.settings.MIN_SERIALIZED_DAG_FETCH_INTERVAL', 5)
    def test_get_dag_with_dag_serialization(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test that Serialized DAG is updated in DagBag when it is updated in\n        Serialized DAG table after 'min_serialized_dag_fetch_interval' seconds are passed.\n        "
        with time_machine.travel(tz.datetime(2020, 1, 5, 0, 0, 0), tick=False):
            example_bash_op_dag = DagBag(include_examples=True).dags.get('example_bash_operator')
            SerializedDagModel.write_dag(dag=example_bash_op_dag)
            dag_bag = DagBag(read_dags_from_db=True)
            ser_dag_1 = dag_bag.get_dag('example_bash_operator')
            ser_dag_1_update_time = dag_bag.dags_last_fetched['example_bash_operator']
            assert example_bash_op_dag.tags == ser_dag_1.tags
            assert ser_dag_1_update_time == tz.datetime(2020, 1, 5, 0, 0, 0)
        with time_machine.travel(tz.datetime(2020, 1, 5, 0, 0, 4), tick=False):
            with assert_queries_count(0):
                assert dag_bag.get_dag('example_bash_operator').tags == ['example', 'example2']
        with time_machine.travel(tz.datetime(2020, 1, 5, 0, 0, 6), tick=False):
            example_bash_op_dag.tags += ['new_tag']
            SerializedDagModel.write_dag(dag=example_bash_op_dag)
        with time_machine.travel(tz.datetime(2020, 1, 5, 0, 0, 8), tick=False):
            with assert_queries_count(2):
                updated_ser_dag_1 = dag_bag.get_dag('example_bash_operator')
                updated_ser_dag_1_update_time = dag_bag.dags_last_fetched['example_bash_operator']
        assert set(updated_ser_dag_1.tags) == {'example', 'example2', 'new_tag'}
        assert updated_ser_dag_1_update_time > ser_dag_1_update_time

    @patch('airflow.models.dagbag.settings.MIN_SERIALIZED_DAG_UPDATE_INTERVAL', 5)
    @patch('airflow.models.dagbag.settings.MIN_SERIALIZED_DAG_FETCH_INTERVAL', 5)
    def test_get_dag_refresh_race_condition(self):
        if False:
            return 10
        '\n        Test that DagBag.get_dag correctly refresh the Serialized DAG even if SerializedDagModel.last_updated\n        is before DagBag.dags_last_fetched.\n        '
        with time_machine.travel(tz.datetime(2020, 1, 5, 0, 0, 0), tick=False):
            example_bash_op_dag = DagBag(include_examples=True).dags.get('example_bash_operator')
            SerializedDagModel.write_dag(dag=example_bash_op_dag)
        with time_machine.travel(tz.datetime(2020, 1, 5, 1, 0, 10), tick=False):
            dag_bag = DagBag(read_dags_from_db=True)
            with assert_queries_count(2):
                ser_dag = dag_bag.get_dag('example_bash_operator')
            ser_dag_update_time = dag_bag.dags_last_fetched['example_bash_operator']
            assert ser_dag.tags == ['example', 'example2']
            assert ser_dag_update_time == tz.datetime(2020, 1, 5, 1, 0, 10)
            with create_session() as session:
                assert SerializedDagModel.get_last_updated_datetime(dag_id='example_bash_operator', session=session) == tz.datetime(2020, 1, 5, 0, 0, 0)
        with time_machine.travel(tz.datetime(2020, 1, 5, 1, 0, 0), tick=False):
            example_bash_op_dag.tags += ['new_tag']
            SerializedDagModel.write_dag(dag=example_bash_op_dag)
        with time_machine.travel(tz.datetime(2020, 1, 5, 1, 0, 30), tick=False):
            with assert_queries_count(2):
                updated_ser_dag = dag_bag.get_dag('example_bash_operator')
                updated_ser_dag_update_time = dag_bag.dags_last_fetched['example_bash_operator']
        assert set(updated_ser_dag.tags) == {'example', 'example2', 'new_tag'}
        assert updated_ser_dag_update_time > ser_dag_update_time

    def test_collect_dags_from_db(self):
        if False:
            for i in range(10):
                print('nop')
        'DAGs are collected from Database'
        db.clear_db_dags()
        dagbag = DagBag(str(example_dags_folder))
        example_dags = dagbag.dags
        for dag in example_dags.values():
            SerializedDagModel.write_dag(dag)
        new_dagbag = DagBag(read_dags_from_db=True)
        assert len(new_dagbag.dags) == 0
        new_dagbag.collect_dags_from_db()
        new_dags = new_dagbag.dags
        assert len(example_dags) == len(new_dags)
        for (dag_id, dag) in example_dags.items():
            serialized_dag = new_dags[dag_id]
            assert serialized_dag.dag_id == dag.dag_id
            assert set(serialized_dag.task_dict) == set(dag.task_dict)

    @patch('airflow.settings.task_policy', cluster_policies.example_task_policy)
    def test_task_cluster_policy_violation(self):
        if False:
            while True:
                i = 10
        '\n        test that file processing results in import error when task does not\n        obey cluster policy.\n        '
        dag_file = os.path.join(TEST_DAGS_FOLDER, 'test_missing_owner.py')
        dag_id = 'test_missing_owner'
        err_cls_name = 'AirflowClusterPolicyViolation'
        dagbag = DagBag(dag_folder=dag_file, include_examples=False)
        assert set() == set(dagbag.dag_ids)
        expected_import_errors = {dag_file: f'{err_cls_name}: DAG policy violation (DAG ID: {dag_id}, Path: {dag_file}):\nNotices:\n * Task must have non-None non-default owner. Current value: airflow'}
        assert expected_import_errors == dagbag.import_errors

    @patch('airflow.settings.task_policy', cluster_policies.example_task_policy)
    def test_task_cluster_policy_nonstring_owner(self):
        if False:
            while True:
                i = 10
        '\n        test that file processing results in import error when task does not\n        obey cluster policy and has owner whose type is not string.\n        '
        TEST_DAGS_CORRUPTED_FOLDER = pathlib.Path(__file__).parent.with_name('dags_corrupted')
        dag_file = os.path.join(TEST_DAGS_CORRUPTED_FOLDER, 'test_nonstring_owner.py')
        dag_id = 'test_nonstring_owner'
        err_cls_name = 'AirflowClusterPolicyViolation'
        dagbag = DagBag(dag_folder=dag_file, include_examples=False)
        assert set() == set(dagbag.dag_ids)
        expected_import_errors = {dag_file: f"{err_cls_name}: DAG policy violation (DAG ID: {dag_id}, Path: {dag_file}):\nNotices:\n * owner should be a string. Current value: ['a']"}
        assert expected_import_errors == dagbag.import_errors

    @patch('airflow.settings.task_policy', cluster_policies.example_task_policy)
    def test_task_cluster_policy_obeyed(self):
        if False:
            while True:
                i = 10
        '\n        test that dag successfully imported without import errors when tasks\n        obey cluster policy.\n        '
        dag_file = os.path.join(TEST_DAGS_FOLDER, 'test_with_non_default_owner.py')
        dagbag = DagBag(dag_folder=dag_file, include_examples=False)
        assert {'test_with_non_default_owner'} == set(dagbag.dag_ids)
        assert {} == dagbag.import_errors

    @patch('airflow.settings.dag_policy', cluster_policies.dag_policy)
    def test_dag_cluster_policy_obeyed(self):
        if False:
            while True:
                i = 10
        dag_file = os.path.join(TEST_DAGS_FOLDER, 'test_dag_with_no_tags.py')
        dagbag = DagBag(dag_folder=dag_file, include_examples=False)
        assert len(dagbag.dag_ids) == 0
        assert 'has no tags' in dagbag.import_errors[dag_file]

    def test_dagbag_dag_collection(self):
        if False:
            while True:
                i = 10
        dagbag = DagBag(dag_folder=TEST_DAGS_FOLDER, include_examples=False, collect_dags=False)
        assert not dagbag.dags
        dagbag.collect_dags()
        assert dagbag.dags
        dagbag = DagBag(dag_folder=TEST_DAGS_FOLDER, include_examples=False)
        assert dagbag.dags