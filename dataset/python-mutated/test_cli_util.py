from __future__ import annotations
import ast
import json
import os
import sys
from argparse import Namespace
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from unittest import mock
import pytest
import airflow
from airflow import settings
from airflow.exceptions import AirflowException
from airflow.models.log import Log
from airflow.utils import cli, cli_action_loggers, timezone
from airflow.utils.cli import _search_for_dag_file, get_dag_by_pickle
repo_root = Path(airflow.__file__).parent.parent

class TestCliUtil:

    def test_metrics_build(self):
        if False:
            i = 10
            return i + 15
        func_name = 'test'
        exec_date = timezone.utcnow()
        namespace = Namespace(dag_id='foo', task_id='bar', subcommand='test', execution_date=exec_date)
        metrics = cli._build_metrics(func_name, namespace)
        expected = {'user': os.environ.get('USER'), 'sub_command': 'test', 'dag_id': 'foo', 'task_id': 'bar', 'execution_date': exec_date}
        for (k, v) in expected.items():
            assert v == metrics.get(k)
        assert metrics.get('start_datetime') <= datetime.utcnow()
        assert metrics.get('full_command')

    def test_fail_function(self):
        if False:
            i = 10
            return i + 15
        '\n        Actual function is failing and fail needs to be propagated.\n        :return:\n        '
        with pytest.raises(NotImplementedError):
            fail_func(Namespace())

    def test_success_function(self):
        if False:
            print('Hello World!')
        '\n        Test success function but with failing callback.\n        In this case, failure should not propagate.\n        :return:\n        '
        with fail_action_logger_callback():
            success_func(Namespace())

    def test_process_subdir_path_with_placeholder(self):
        if False:
            for i in range(10):
                print('nop')
        assert os.path.join(settings.DAGS_FOLDER, 'abc') == cli.process_subdir('DAGS_FOLDER/abc')

    @pytest.mark.db_test
    def test_get_dags(self):
        if False:
            return 10
        dags = cli.get_dags(None, 'example_subdag_operator')
        assert len(dags) == 1
        dags = cli.get_dags(None, 'subdag', True)
        assert len(dags) > 1
        with pytest.raises(AirflowException):
            cli.get_dags(None, 'foobar', True)

    @pytest.mark.db_test
    @pytest.mark.parametrize(['given_command', 'expected_masked_command'], [('airflow users create -u test2 -l doe -f jon -e jdoe@apache.org -r admin --password test', 'airflow users create -u test2 -l doe -f jon -e jdoe@apache.org -r admin --password ********'), ('airflow users create -u test2 -l doe -f jon -e jdoe@apache.org -r admin -p test', 'airflow users create -u test2 -l doe -f jon -e jdoe@apache.org -r admin -p ********'), ('airflow users create -u test2 -l doe -f jon -e jdoe@apache.org -r admin --password=test', 'airflow users create -u test2 -l doe -f jon -e jdoe@apache.org -r admin --password=********'), ('airflow users create -u test2 -l doe -f jon -e jdoe@apache.org -r admin -p=test', 'airflow users create -u test2 -l doe -f jon -e jdoe@apache.org -r admin -p=********'), ('airflow connections add dsfs --conn-login asd --conn-password test --conn-type google', 'airflow connections add dsfs --conn-login asd --conn-password ******** --conn-type google'), ('airflow scheduler -p', 'airflow scheduler -p'), ('airflow celery flower -p 8888', 'airflow celery flower -p 8888')])
    def test_cli_create_user_supplied_password_is_masked(self, given_command, expected_masked_command, session):
        if False:
            print('Hello World!')
        args = given_command.split()
        expected_command = expected_masked_command.split()
        exec_date = timezone.utcnow()
        namespace = Namespace(dag_id='foo', task_id='bar', subcommand='test', execution_date=exec_date)
        with mock.patch.object(sys, 'argv', args), mock.patch('airflow.utils.session.create_session') as mock_create_session:
            metrics = cli._build_metrics(args[1], namespace)
            mock_create_session.return_value = session.begin_nested()
            mock_create_session.return_value.bulk_insert_mappings = session.bulk_insert_mappings
            cli_action_loggers.default_action_log(**metrics)
            log = session.query(Log).order_by(Log.dttm.desc()).first()
        assert metrics.get('start_datetime') <= datetime.utcnow()
        command: str = json.loads(log.extra).get('full_command')
        command = ast.literal_eval(command)
        assert command == expected_command

    def test_setup_locations_relative_pid_path(self):
        if False:
            print('Hello World!')
        relative_pid_path = 'fake.pid'
        pid_full_path = os.path.join(os.getcwd(), relative_pid_path)
        (pid, _, _, _) = cli.setup_locations(process='fake_process', pid=relative_pid_path)
        assert pid == pid_full_path

    def test_setup_locations_absolute_pid_path(self):
        if False:
            print('Hello World!')
        abs_pid_path = os.path.join(os.getcwd(), 'fake.pid')
        (pid, _, _, _) = cli.setup_locations(process='fake_process', pid=abs_pid_path)
        assert pid == abs_pid_path

    def test_setup_locations_none_pid_path(self):
        if False:
            print('Hello World!')
        process_name = 'fake_process'
        default_pid_path = os.path.join(settings.AIRFLOW_HOME, f'airflow-{process_name}.pid')
        (pid, _, _, _) = cli.setup_locations(process=process_name)
        assert pid == default_pid_path

    @pytest.mark.db_test
    def test_get_dag_by_pickle(self, session, dag_maker):
        if False:
            return 10
        from airflow.models.dagpickle import DagPickle
        with dag_maker(dag_id='test_get_dag_by_pickle') as dag:
            pass
        dp = DagPickle(dag=dag)
        session.add(dp)
        session.commit()
        dp_from_db = get_dag_by_pickle(pickle_id=dp.id, session=session)
        assert dp_from_db.dag_id == 'test_get_dag_by_pickle'
        with pytest.raises(AirflowException, match='pickle_id could not be found .* -42'):
            get_dag_by_pickle(pickle_id=-42, session=session)

@contextmanager
def fail_action_logger_callback():
    if False:
        return 10
    'Adding failing callback and revert it back when closed.'
    tmp = cli_action_loggers.__pre_exec_callbacks[:]

    def fail_callback(**_):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError
    cli_action_loggers.register_pre_exec_callback(fail_callback)
    yield
    cli_action_loggers.__pre_exec_callbacks = tmp

@cli.action_cli(check_db=False)
def fail_func(_):
    if False:
        while True:
            i = 10
    raise NotImplementedError

@cli.action_cli(check_db=False)
def success_func(_):
    if False:
        i = 10
        return i + 15
    pass

def test__search_for_dags_file():
    if False:
        for i in range(10):
            print('nop')
    dags_folder = settings.DAGS_FOLDER
    assert _search_for_dag_file('') is None
    assert _search_for_dag_file(None) is None
    assert _search_for_dag_file('any/hi/test_dags_folder.py') == str(Path(dags_folder) / 'test_dags_folder.py')
    existing_folder = Path(settings.DAGS_FOLDER, 'subdir1')
    assert existing_folder.exists()
    assert _search_for_dag_file(existing_folder.as_posix()) is None
    assert _search_for_dag_file('any/hi/__init__.py') is None