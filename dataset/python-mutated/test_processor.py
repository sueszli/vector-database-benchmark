from __future__ import annotations
import datetime
import os
from unittest import mock
from unittest.mock import MagicMock, patch
from zipfile import ZipFile
import pytest
from airflow import PY311, settings
from airflow.callbacks.callback_requests import TaskCallbackRequest
from airflow.configuration import TEST_DAGS_FOLDER, conf
from airflow.dag_processing.manager import DagFileProcessorAgent
from airflow.dag_processing.processor import DagFileProcessor, DagFileProcessorProcess
from airflow.models import DagBag, DagModel, SlaMiss, TaskInstance, errors
from airflow.models.serialized_dag import SerializedDagModel
from airflow.models.taskinstance import SimpleTaskInstance
from airflow.operators.empty import EmptyOperator
from airflow.utils import timezone
from airflow.utils.session import create_session
from airflow.utils.state import State
from airflow.utils.types import DagRunType
from tests.test_utils.config import conf_vars, env_vars
from tests.test_utils.db import clear_db_dags, clear_db_import_errors, clear_db_jobs, clear_db_pools, clear_db_runs, clear_db_serialized_dags, clear_db_sla_miss
from tests.test_utils.mock_executor import MockExecutor
pytestmark = pytest.mark.db_test
DEFAULT_DATE = timezone.datetime(2016, 1, 1)
PARSEABLE_DAG_FILE_CONTENTS = '"airflow DAG"'
UNPARSEABLE_DAG_FILE_CONTENTS = 'airflow DAG'
INVALID_DAG_WITH_DEPTH_FILE_CONTENTS = 'def something():\n    return airflow_DAG\nsomething()'
TEMP_DAG_FILENAME = 'temp_dag.py'

@pytest.fixture(scope='class')
def disable_load_example():
    if False:
        while True:
            i = 10
    with conf_vars({('core', 'load_examples'): 'false'}):
        with env_vars({'AIRFLOW__CORE__LOAD_EXAMPLES': 'false'}):
            yield

@pytest.mark.usefixtures('disable_load_example')
class TestDagFileProcessor:

    @staticmethod
    def clean_db():
        if False:
            i = 10
            return i + 15
        clear_db_runs()
        clear_db_pools()
        clear_db_dags()
        clear_db_sla_miss()
        clear_db_import_errors()
        clear_db_jobs()
        clear_db_serialized_dags()

    def setup_class(self):
        if False:
            for i in range(10):
                print('nop')
        self.clean_db()

    def setup_method(self):
        if False:
            i = 10
            return i + 15
        self.null_exec = MockExecutor()
        self.scheduler_job = None

    def teardown_method(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self.scheduler_job and self.scheduler_job.job_runner.processor_agent:
            self.scheduler_job.job_runner.processor_agent.end()
            self.scheduler_job = None
        self.clean_db()

    def _process_file(self, file_path, dag_directory, session):
        if False:
            return 10
        dag_file_processor = DagFileProcessor(dag_ids=[], dag_directory=str(dag_directory), log=mock.MagicMock())
        dag_file_processor.process_file(file_path, [], False, session)

    @mock.patch('airflow.dag_processing.processor.DagFileProcessor._get_dagbag')
    def test_dag_file_processor_sla_miss_callback(self, mock_get_dagbag, create_dummy_dag, get_test_dag):
        if False:
            return 10
        '\n        Test that the dag file processor calls the sla miss callback\n        '
        session = settings.Session()
        sla_callback = MagicMock()
        test_start_date = timezone.utcnow() - datetime.timedelta(days=1)
        (dag, task) = create_dummy_dag(dag_id='test_sla_miss', task_id='dummy', sla_miss_callback=sla_callback, default_args={'start_date': test_start_date, 'sla': datetime.timedelta()})
        session.merge(TaskInstance(task=task, execution_date=test_start_date, state='success'))
        session.merge(SlaMiss(task_id='dummy', dag_id='test_sla_miss', execution_date=test_start_date))
        mock_dagbag = mock.Mock()
        mock_dagbag.get_dag.return_value = dag
        mock_get_dagbag.return_value = mock_dagbag
        DagFileProcessor.manage_slas(dag_folder=dag.fileloc, dag_id='test_sla_miss', session=session)
        assert sla_callback.called

    @mock.patch('airflow.dag_processing.processor.DagFileProcessor._get_dagbag')
    def test_dag_file_processor_sla_miss_callback_invalid_sla(self, mock_get_dagbag, create_dummy_dag):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that the dag file processor does not call the sla miss callback when\n        given an invalid sla\n        '
        session = settings.Session()
        sla_callback = MagicMock()
        test_start_date = timezone.utcnow() - datetime.timedelta(days=1)
        (dag, task) = create_dummy_dag(dag_id='test_sla_miss', task_id='dummy', sla_miss_callback=sla_callback, default_args={'start_date': test_start_date, 'sla': None})
        session.merge(TaskInstance(task=task, execution_date=test_start_date, state='success'))
        session.merge(SlaMiss(task_id='dummy', dag_id='test_sla_miss', execution_date=test_start_date))
        mock_dagbag = mock.Mock()
        mock_dagbag.get_dag.return_value = dag
        mock_get_dagbag.return_value = mock_dagbag
        DagFileProcessor.manage_slas(dag_folder=dag.fileloc, dag_id='test_sla_miss', session=session)
        sla_callback.assert_not_called()

    @mock.patch('airflow.dag_processing.processor.DagFileProcessor._get_dagbag')
    def test_dag_file_processor_sla_miss_callback_sent_notification(self, mock_get_dagbag, create_dummy_dag):
        if False:
            return 10
        '\n        Test that the dag file processor does not call the sla_miss_callback when a\n        notification has already been sent\n        '
        session = settings.Session()
        sla_callback = MagicMock()
        test_start_date = timezone.utcnow() - datetime.timedelta(days=2)
        (dag, task) = create_dummy_dag(dag_id='test_sla_miss', task_id='dummy', sla_miss_callback=sla_callback, default_args={'start_date': test_start_date, 'sla': datetime.timedelta(days=1)})
        session.merge(TaskInstance(task=task, execution_date=test_start_date, state='success'))
        session.merge(SlaMiss(task_id='dummy', dag_id='test_sla_miss', execution_date=test_start_date, email_sent=False, notification_sent=True))
        mock_dagbag = mock.Mock()
        mock_dagbag.get_dag.return_value = dag
        mock_get_dagbag.return_value = mock_dagbag
        DagFileProcessor.manage_slas(dag_folder=dag.fileloc, dag_id='test_sla_miss', session=session)
        sla_callback.assert_not_called()

    @mock.patch('airflow.dag_processing.processor.Stats.incr')
    @mock.patch('airflow.dag_processing.processor.DagFileProcessor._get_dagbag')
    def test_dag_file_processor_sla_miss_doesnot_raise_integrity_error(self, mock_get_dagbag, mock_stats_incr, dag_maker):
        if False:
            return 10
        '\n        Test that the dag file processor does not try to insert already existing item into the database\n        '
        session = settings.Session()
        test_start_date = timezone.utcnow() - datetime.timedelta(days=2)
        with dag_maker(dag_id='test_sla_miss', default_args={'start_date': test_start_date, 'sla': datetime.timedelta(days=1)}) as dag:
            task = EmptyOperator(task_id='dummy')
        dag_maker.create_dagrun(execution_date=test_start_date, state=State.SUCCESS)
        ti = TaskInstance(task=task, execution_date=test_start_date, state='success')
        session.merge(ti)
        session.flush()
        mock_dagbag = mock.Mock()
        mock_dagbag.get_dag.return_value = dag
        mock_get_dagbag.return_value = mock_dagbag
        DagFileProcessor.manage_slas(dag_folder=dag.fileloc, dag_id='test_sla_miss', session=session)
        sla_miss_count = session.query(SlaMiss).filter(SlaMiss.dag_id == dag.dag_id, SlaMiss.task_id == task.task_id).count()
        assert sla_miss_count == 1
        mock_stats_incr.assert_called_with('sla_missed', tags={'dag_id': 'test_sla_miss', 'task_id': 'dummy'})
        DagFileProcessor.manage_slas(dag_folder=dag.fileloc, dag_id='test_sla_miss', session=session)

    @mock.patch('airflow.dag_processing.processor.Stats.incr')
    @mock.patch('airflow.dag_processing.processor.DagFileProcessor._get_dagbag')
    def test_dag_file_processor_sla_miss_continue_checking_the_task_instances_after_recording_missing_sla(self, mock_get_dagbag, mock_stats_incr, dag_maker):
        if False:
            print('Hello World!')
        '\n        Test that the dag file processor continue checking subsequent task instances\n        even if the preceding task instance misses the sla ahead\n        '
        session = settings.Session()
        now = timezone.utcnow()
        test_start_date = now - datetime.timedelta(days=3)
        with dag_maker(dag_id='test_sla_miss', default_args={'start_date': test_start_date, 'sla': datetime.timedelta(days=1)}) as dag:
            task = EmptyOperator(task_id='dummy')
        dag_maker.create_dagrun(execution_date=test_start_date, state=State.SUCCESS)
        session.merge(TaskInstance(task=task, execution_date=test_start_date, state='success'))
        session.merge(SlaMiss(task_id=task.task_id, dag_id=dag.dag_id, execution_date=now - datetime.timedelta(days=2)))
        session.flush()
        mock_dagbag = mock.Mock()
        mock_dagbag.get_dag.return_value = dag
        mock_get_dagbag.return_value = mock_dagbag
        DagFileProcessor.manage_slas(dag_folder=dag.fileloc, dag_id='test_sla_miss', session=session)
        sla_miss_count = session.query(SlaMiss).filter(SlaMiss.dag_id == dag.dag_id, SlaMiss.task_id == task.task_id).count()
        assert sla_miss_count == 2
        mock_stats_incr.assert_called_with('sla_missed', tags={'dag_id': 'test_sla_miss', 'task_id': 'dummy'})

    @patch.object(DagFileProcessor, 'logger')
    @mock.patch('airflow.dag_processing.processor.Stats.incr')
    @mock.patch('airflow.dag_processing.processor.DagFileProcessor._get_dagbag')
    def test_dag_file_processor_sla_miss_callback_exception(self, mock_get_dagbag, mock_stats_incr, mock_get_log, create_dummy_dag):
        if False:
            print('Hello World!')
        '\n        Test that the dag file processor gracefully logs an exception if there is a problem\n        calling the sla_miss_callback\n        '
        session = settings.Session()
        sla_callback = MagicMock(__name__='function_name', side_effect=RuntimeError('Could not call function'))
        test_start_date = timezone.utcnow() - datetime.timedelta(days=1)
        for (i, callback) in enumerate([[sla_callback], sla_callback]):
            (dag, task) = create_dummy_dag(dag_id=f'test_sla_miss_{i}', task_id='dummy', sla_miss_callback=callback, default_args={'start_date': test_start_date, 'sla': datetime.timedelta(hours=1)})
            mock_stats_incr.reset_mock()
            session.merge(TaskInstance(task=task, execution_date=test_start_date, state='Success'))
            session.merge(SlaMiss(task_id='dummy', dag_id=f'test_sla_miss_{i}', execution_date=test_start_date))
            mock_log = mock.Mock()
            mock_get_log.return_value = mock_log
            mock_dagbag = mock.Mock()
            mock_dagbag.get_dag.return_value = dag
            mock_get_dagbag.return_value = mock_dagbag
            DagFileProcessor.manage_slas(dag_folder=dag.fileloc, dag_id='test_sla_miss', session=session)
            assert sla_callback.called
            mock_log.exception.assert_called_once_with('Could not call sla_miss_callback(%s) for DAG %s', sla_callback.__name__, f'test_sla_miss_{i}')
            mock_stats_incr.assert_called_once_with('sla_callback_notification_failure', tags={'dag_id': f'test_sla_miss_{i}', 'func_name': sla_callback.__name__})

    @mock.patch('airflow.dag_processing.processor.send_email')
    @mock.patch('airflow.dag_processing.processor.DagFileProcessor._get_dagbag')
    def test_dag_file_processor_only_collect_emails_from_sla_missed_tasks(self, mock_get_dagbag, mock_send_email, create_dummy_dag):
        if False:
            while True:
                i = 10
        session = settings.Session()
        test_start_date = timezone.utcnow() - datetime.timedelta(days=1)
        email1 = 'test1@test.com'
        (dag, task) = create_dummy_dag(dag_id='test_sla_miss', task_id='sla_missed', email=email1, default_args={'start_date': test_start_date, 'sla': datetime.timedelta(hours=1)})
        session.merge(TaskInstance(task=task, execution_date=test_start_date, state='Success'))
        email2 = 'test2@test.com'
        EmptyOperator(task_id='sla_not_missed', dag=dag, owner='airflow', email=email2)
        session.merge(SlaMiss(task_id='sla_missed', dag_id='test_sla_miss', execution_date=test_start_date))
        mock_dagbag = mock.Mock()
        mock_dagbag.get_dag.return_value = dag
        mock_get_dagbag.return_value = mock_dagbag
        DagFileProcessor.manage_slas(dag_folder=dag.fileloc, dag_id='test_sla_miss', session=session)
        assert len(mock_send_email.call_args_list) == 1
        send_email_to = mock_send_email.call_args_list[0][0][0]
        assert email1 in send_email_to
        assert email2 not in send_email_to

    @patch.object(DagFileProcessor, 'logger')
    @mock.patch('airflow.dag_processing.processor.Stats.incr')
    @mock.patch('airflow.utils.email.send_email')
    @mock.patch('airflow.dag_processing.processor.DagFileProcessor._get_dagbag')
    def test_dag_file_processor_sla_miss_email_exception(self, mock_get_dagbag, mock_send_email, mock_stats_incr, mock_get_log, create_dummy_dag):
        if False:
            print('Hello World!')
        '\n        Test that the dag file processor gracefully logs an exception if there is a problem\n        sending an email\n        '
        session = settings.Session()
        dag_id = 'test_sla_miss'
        task_id = 'test_ti'
        email = 'test@test.com'
        mock_send_email.side_effect = RuntimeError('Could not send an email')
        test_start_date = timezone.utcnow() - datetime.timedelta(days=1)
        (dag, task) = create_dummy_dag(dag_id=dag_id, task_id=task_id, email=email, default_args={'start_date': test_start_date, 'sla': datetime.timedelta(hours=1)})
        mock_stats_incr.reset_mock()
        session.merge(TaskInstance(task=task, execution_date=test_start_date, state='Success'))
        session.merge(SlaMiss(task_id=task_id, dag_id=dag_id, execution_date=test_start_date))
        mock_log = mock.Mock()
        mock_get_log.return_value = mock_log
        mock_dagbag = mock.Mock()
        mock_dagbag.get_dag.return_value = dag
        mock_get_dagbag.return_value = mock_dagbag
        DagFileProcessor.manage_slas(dag_folder=dag.fileloc, dag_id=dag_id, session=session)
        mock_log.exception.assert_called_once_with('Could not send SLA Miss email notification for DAG %s', dag_id)
        mock_stats_incr.assert_called_once_with('sla_email_notification_failure', tags={'dag_id': dag_id})

    @mock.patch('airflow.dag_processing.processor.DagFileProcessor._get_dagbag')
    def test_dag_file_processor_sla_miss_deleted_task(self, mock_get_dagbag, create_dummy_dag):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that the dag file processor will not crash when trying to send\n        sla miss notification for a deleted task\n        '
        session = settings.Session()
        test_start_date = timezone.utcnow() - datetime.timedelta(days=1)
        (dag, task) = create_dummy_dag(dag_id='test_sla_miss', task_id='dummy', email='test@test.com', default_args={'start_date': test_start_date, 'sla': datetime.timedelta(hours=1)})
        session.merge(TaskInstance(task=task, execution_date=test_start_date, state='Success'))
        session.merge(SlaMiss(task_id='dummy_deleted', dag_id='test_sla_miss', execution_date=test_start_date))
        mock_dagbag = mock.Mock()
        mock_dagbag.get_dag.return_value = dag
        mock_get_dagbag.return_value = mock_dagbag
        DagFileProcessor.manage_slas(dag_folder=dag.fileloc, dag_id='test_sla_miss', session=session)

    @patch.object(TaskInstance, 'handle_failure')
    def test_execute_on_failure_callbacks(self, mock_ti_handle_failure):
        if False:
            i = 10
            return i + 15
        dagbag = DagBag(dag_folder='/dev/null', include_examples=True, read_dags_from_db=False)
        dag_file_processor = DagFileProcessor(dag_ids=[], dag_directory=TEST_DAGS_FOLDER, log=mock.MagicMock())
        with create_session() as session:
            session.query(TaskInstance).delete()
            dag = dagbag.get_dag('example_branch_operator')
            dagrun = dag.create_dagrun(state=State.RUNNING, execution_date=DEFAULT_DATE, run_type=DagRunType.SCHEDULED, session=session)
            task = dag.get_task(task_id='run_this_first')
            ti = TaskInstance(task, run_id=dagrun.run_id, state=State.RUNNING)
            session.add(ti)
        requests = [TaskCallbackRequest(full_filepath='A', simple_task_instance=SimpleTaskInstance.from_ti(ti), msg='Message')]
        dag_file_processor.execute_callbacks(dagbag, requests, session)
        mock_ti_handle_failure.assert_called_once_with(error='Message', test_mode=conf.getboolean('core', 'unit_test_mode'), session=session)

    @pytest.mark.parametrize(['has_serialized_dag'], [pytest.param(True, id='dag_in_db'), pytest.param(False, id='no_dag_found')])
    @patch.object(TaskInstance, 'handle_failure')
    def test_execute_on_failure_callbacks_without_dag(self, mock_ti_handle_failure, has_serialized_dag):
        if False:
            i = 10
            return i + 15
        dagbag = DagBag(dag_folder='/dev/null', include_examples=True, read_dags_from_db=False)
        dag_file_processor = DagFileProcessor(dag_ids=[], dag_directory=TEST_DAGS_FOLDER, log=mock.MagicMock())
        with create_session() as session:
            session.query(TaskInstance).delete()
            dag = dagbag.get_dag('example_branch_operator')
            dagrun = dag.create_dagrun(state=State.RUNNING, execution_date=DEFAULT_DATE, run_type=DagRunType.SCHEDULED, session=session)
            task = dag.get_task(task_id='run_this_first')
            ti = TaskInstance(task, run_id=dagrun.run_id, state=State.QUEUED)
            session.add(ti)
            if has_serialized_dag:
                assert SerializedDagModel.write_dag(dag, session=session) is True
                session.flush()
        requests = [TaskCallbackRequest(full_filepath='A', simple_task_instance=SimpleTaskInstance.from_ti(ti), msg='Message')]
        dag_file_processor.execute_callbacks_without_dag(requests, session)
        mock_ti_handle_failure.assert_called_once_with(error='Message', test_mode=conf.getboolean('core', 'unit_test_mode'), session=session)

    def test_failure_callbacks_should_not_drop_hostname(self):
        if False:
            i = 10
            return i + 15
        dagbag = DagBag(dag_folder='/dev/null', include_examples=True, read_dags_from_db=False)
        dag_file_processor = DagFileProcessor(dag_ids=[], dag_directory=TEST_DAGS_FOLDER, log=mock.MagicMock())
        dag_file_processor.UNIT_TEST_MODE = False
        with create_session() as session:
            dag = dagbag.get_dag('example_branch_operator')
            task = dag.get_task(task_id='run_this_first')
            dagrun = dag.create_dagrun(state=State.RUNNING, execution_date=DEFAULT_DATE, run_type=DagRunType.SCHEDULED, session=session)
            ti = TaskInstance(task, run_id=dagrun.run_id, state=State.RUNNING)
            ti.hostname = 'test_hostname'
            session.add(ti)
        requests = [TaskCallbackRequest(full_filepath='A', simple_task_instance=SimpleTaskInstance.from_ti(ti), msg='Message')]
        dag_file_processor.execute_callbacks(dagbag, requests)
        with create_session() as session:
            tis = session.query(TaskInstance)
            assert tis[0].hostname == 'test_hostname'

    def test_process_file_should_failure_callback(self, monkeypatch, tmp_path, get_test_dag):
        if False:
            return 10
        callback_file = tmp_path.joinpath('callback.txt')
        callback_file.touch()
        monkeypatch.setenv('AIRFLOW_CALLBACK_FILE', str(callback_file))
        dag_file_processor = DagFileProcessor(dag_ids=[], dag_directory=TEST_DAGS_FOLDER, log=mock.MagicMock())
        dag = get_test_dag('test_on_failure_callback')
        task = dag.get_task(task_id='test_on_failure_callback_task')
        with create_session() as session:
            dagrun = dag.create_dagrun(state=State.RUNNING, execution_date=DEFAULT_DATE, run_type=DagRunType.SCHEDULED, session=session)
            ti = dagrun.get_task_instance(task.task_id)
            ti.refresh_from_task(task)
            requests = [TaskCallbackRequest(full_filepath=dag.fileloc, simple_task_instance=SimpleTaskInstance.from_ti(ti), msg='Message')]
            dag_file_processor.process_file(dag.fileloc, requests, session=session)
        ti.refresh_from_db()
        msg = ' '.join([str(k) for k in ti.key.primary]) + ' fired callback'
        assert msg in callback_file.read_text()

    @conf_vars({('core', 'dagbag_import_error_tracebacks'): 'False'})
    def test_add_unparseable_file_before_sched_start_creates_import_error(self, tmpdir):
        if False:
            for i in range(10):
                print('nop')
        unparseable_filename = os.path.join(tmpdir, TEMP_DAG_FILENAME)
        with open(unparseable_filename, 'w') as unparseable_file:
            unparseable_file.writelines(UNPARSEABLE_DAG_FILE_CONTENTS)
        with create_session() as session:
            self._process_file(unparseable_filename, dag_directory=tmpdir, session=session)
            import_errors = session.query(errors.ImportError).all()
            assert len(import_errors) == 1
            import_error = import_errors[0]
            assert import_error.filename == unparseable_filename
            assert import_error.stacktrace == f'invalid syntax ({TEMP_DAG_FILENAME}, line 1)'
            session.rollback()

    @conf_vars({('core', 'dagbag_import_error_tracebacks'): 'False'})
    def test_add_unparseable_zip_file_creates_import_error(self, tmpdir):
        if False:
            print('Hello World!')
        zip_filename = os.path.join(tmpdir, 'test_zip.zip')
        invalid_dag_filename = os.path.join(zip_filename, TEMP_DAG_FILENAME)
        with ZipFile(zip_filename, 'w') as zip_file:
            zip_file.writestr(TEMP_DAG_FILENAME, UNPARSEABLE_DAG_FILE_CONTENTS)
        with create_session() as session:
            self._process_file(zip_filename, dag_directory=tmpdir, session=session)
            import_errors = session.query(errors.ImportError).all()
            assert len(import_errors) == 1
            import_error = import_errors[0]
            assert import_error.filename == invalid_dag_filename
            assert import_error.stacktrace == f'invalid syntax ({TEMP_DAG_FILENAME}, line 1)'
            session.rollback()

    @conf_vars({('core', 'dagbag_import_error_tracebacks'): 'False'})
    def test_dag_model_has_import_error_is_true_when_import_error_exists(self, tmpdir, session):
        if False:
            return 10
        dag_file = os.path.join(TEST_DAGS_FOLDER, 'test_example_bash_operator.py')
        temp_dagfile = os.path.join(tmpdir, TEMP_DAG_FILENAME)
        with open(dag_file) as main_dag, open(temp_dagfile, 'w') as next_dag:
            for line in main_dag:
                next_dag.write(line)
        self._process_file(temp_dagfile, dag_directory=tmpdir, session=session)
        dm = session.query(DagModel).filter(DagModel.fileloc == temp_dagfile).first()
        assert not dm.has_import_errors
        with open(temp_dagfile, 'a') as file:
            file.writelines(UNPARSEABLE_DAG_FILE_CONTENTS)
        self._process_file(temp_dagfile, dag_directory=tmpdir, session=session)
        import_errors = session.query(errors.ImportError).all()
        assert len(import_errors) == 1
        import_error = import_errors[0]
        assert import_error.filename == temp_dagfile
        assert import_error.stacktrace
        dm = session.query(DagModel).filter(DagModel.fileloc == temp_dagfile).first()
        assert dm.has_import_errors

    def test_no_import_errors_with_parseable_dag(self, tmpdir):
        if False:
            print('Hello World!')
        parseable_filename = os.path.join(tmpdir, TEMP_DAG_FILENAME)
        with open(parseable_filename, 'w') as parseable_file:
            parseable_file.writelines(PARSEABLE_DAG_FILE_CONTENTS)
        with create_session() as session:
            self._process_file(parseable_filename, dag_directory=tmpdir, session=session)
            import_errors = session.query(errors.ImportError).all()
            assert len(import_errors) == 0
            session.rollback()

    def test_no_import_errors_with_parseable_dag_in_zip(self, tmpdir):
        if False:
            i = 10
            return i + 15
        zip_filename = os.path.join(tmpdir, 'test_zip.zip')
        with ZipFile(zip_filename, 'w') as zip_file:
            zip_file.writestr(TEMP_DAG_FILENAME, PARSEABLE_DAG_FILE_CONTENTS)
        with create_session() as session:
            self._process_file(zip_filename, dag_directory=tmpdir, session=session)
            import_errors = session.query(errors.ImportError).all()
            assert len(import_errors) == 0
            session.rollback()

    @conf_vars({('core', 'dagbag_import_error_tracebacks'): 'False'})
    def test_new_import_error_replaces_old(self, tmpdir):
        if False:
            print('Hello World!')
        unparseable_filename = os.path.join(tmpdir, TEMP_DAG_FILENAME)
        with open(unparseable_filename, 'w') as unparseable_file:
            unparseable_file.writelines(UNPARSEABLE_DAG_FILE_CONTENTS)
        session = settings.Session()
        self._process_file(unparseable_filename, dag_directory=tmpdir, session=session)
        with open(unparseable_filename, 'w') as unparseable_file:
            unparseable_file.writelines(PARSEABLE_DAG_FILE_CONTENTS + os.linesep + UNPARSEABLE_DAG_FILE_CONTENTS)
        self._process_file(unparseable_filename, dag_directory=tmpdir, session=session)
        import_errors = session.query(errors.ImportError).all()
        assert len(import_errors) == 1
        import_error = import_errors[0]
        assert import_error.filename == unparseable_filename
        assert import_error.stacktrace == f'invalid syntax ({TEMP_DAG_FILENAME}, line 2)'
        session.rollback()

    def test_import_error_record_is_updated_not_deleted_and_recreated(self, tmpdir):
        if False:
            i = 10
            return i + 15
        '\n        Test that existing import error is updated and new record not created\n        for a dag with the same filename\n        '
        filename_to_parse = os.path.join(tmpdir, TEMP_DAG_FILENAME)
        with open(filename_to_parse, 'w') as file_to_parse:
            file_to_parse.writelines(UNPARSEABLE_DAG_FILE_CONTENTS)
        session = settings.Session()
        self._process_file(filename_to_parse, dag_directory=tmpdir, session=session)
        import_error_1 = session.query(errors.ImportError).filter(errors.ImportError.filename == filename_to_parse).one()
        for _ in range(10):
            self._process_file(filename_to_parse, dag_directory=tmpdir, session=session)
        import_error_2 = session.query(errors.ImportError).filter(errors.ImportError.filename == filename_to_parse).one()
        assert import_error_1.id == import_error_2.id

    def test_remove_error_clears_import_error(self, tmpdir):
        if False:
            print('Hello World!')
        filename_to_parse = os.path.join(tmpdir, TEMP_DAG_FILENAME)
        with open(filename_to_parse, 'w') as file_to_parse:
            file_to_parse.writelines(UNPARSEABLE_DAG_FILE_CONTENTS)
        session = settings.Session()
        self._process_file(filename_to_parse, dag_directory=tmpdir, session=session)
        with open(filename_to_parse, 'w') as file_to_parse:
            file_to_parse.writelines(PARSEABLE_DAG_FILE_CONTENTS)
        self._process_file(filename_to_parse, dag_directory=tmpdir, session=session)
        import_errors = session.query(errors.ImportError).all()
        assert len(import_errors) == 0
        session.rollback()

    def test_remove_error_clears_import_error_zip(self, tmpdir):
        if False:
            return 10
        session = settings.Session()
        zip_filename = os.path.join(tmpdir, 'test_zip.zip')
        with ZipFile(zip_filename, 'w') as zip_file:
            zip_file.writestr(TEMP_DAG_FILENAME, UNPARSEABLE_DAG_FILE_CONTENTS)
        self._process_file(zip_filename, dag_directory=tmpdir, session=session)
        import_errors = session.query(errors.ImportError).all()
        assert len(import_errors) == 1
        with ZipFile(zip_filename, 'w') as zip_file:
            zip_file.writestr(TEMP_DAG_FILENAME, 'import os # airflow DAG')
        self._process_file(zip_filename, dag_directory=tmpdir, session=session)
        import_errors = session.query(errors.ImportError).all()
        assert len(import_errors) == 0
        session.rollback()

    def test_import_error_tracebacks(self, tmpdir):
        if False:
            i = 10
            return i + 15
        unparseable_filename = os.path.join(tmpdir, TEMP_DAG_FILENAME)
        with open(unparseable_filename, 'w') as unparseable_file:
            unparseable_file.writelines(INVALID_DAG_WITH_DEPTH_FILE_CONTENTS)
        with create_session() as session:
            self._process_file(unparseable_filename, dag_directory=tmpdir, session=session)
            import_errors = session.query(errors.ImportError).all()
            assert len(import_errors) == 1
            import_error = import_errors[0]
            assert import_error.filename == unparseable_filename
            if PY311:
                expected_stacktrace = 'Traceback (most recent call last):\n  File "{}", line 3, in <module>\n    something()\n  File "{}", line 2, in something\n    return airflow_DAG\n           ^^^^^^^^^^^\nNameError: name \'airflow_DAG\' is not defined\n'
            else:
                expected_stacktrace = 'Traceback (most recent call last):\n  File "{}", line 3, in <module>\n    something()\n  File "{}", line 2, in something\n    return airflow_DAG\nNameError: name \'airflow_DAG\' is not defined\n'
            assert import_error.stacktrace == expected_stacktrace.format(unparseable_filename, unparseable_filename)
            session.rollback()

    @conf_vars({('core', 'dagbag_import_error_traceback_depth'): '1'})
    def test_import_error_traceback_depth(self, tmpdir):
        if False:
            i = 10
            return i + 15
        unparseable_filename = os.path.join(tmpdir, TEMP_DAG_FILENAME)
        with open(unparseable_filename, 'w') as unparseable_file:
            unparseable_file.writelines(INVALID_DAG_WITH_DEPTH_FILE_CONTENTS)
        with create_session() as session:
            self._process_file(unparseable_filename, dag_directory=tmpdir, session=session)
            import_errors = session.query(errors.ImportError).all()
            assert len(import_errors) == 1
            import_error = import_errors[0]
            assert import_error.filename == unparseable_filename
            if PY311:
                expected_stacktrace = 'Traceback (most recent call last):\n  File "{}", line 2, in something\n    return airflow_DAG\n           ^^^^^^^^^^^\nNameError: name \'airflow_DAG\' is not defined\n'
            else:
                expected_stacktrace = 'Traceback (most recent call last):\n  File "{}", line 2, in something\n    return airflow_DAG\nNameError: name \'airflow_DAG\' is not defined\n'
            assert import_error.stacktrace == expected_stacktrace.format(unparseable_filename)
            session.rollback()

    def test_import_error_tracebacks_zip(self, tmpdir):
        if False:
            while True:
                i = 10
        invalid_zip_filename = os.path.join(tmpdir, 'test_zip_invalid.zip')
        invalid_dag_filename = os.path.join(invalid_zip_filename, TEMP_DAG_FILENAME)
        with ZipFile(invalid_zip_filename, 'w') as invalid_zip_file:
            invalid_zip_file.writestr(TEMP_DAG_FILENAME, INVALID_DAG_WITH_DEPTH_FILE_CONTENTS)
        with create_session() as session:
            self._process_file(invalid_zip_filename, dag_directory=tmpdir, session=session)
            import_errors = session.query(errors.ImportError).all()
            assert len(import_errors) == 1
            import_error = import_errors[0]
            assert import_error.filename == invalid_dag_filename
            if PY311:
                expected_stacktrace = 'Traceback (most recent call last):\n  File "{}", line 3, in <module>\n    something()\n  File "{}", line 2, in something\n    return airflow_DAG\n           ^^^^^^^^^^^\nNameError: name \'airflow_DAG\' is not defined\n'
            else:
                expected_stacktrace = 'Traceback (most recent call last):\n  File "{}", line 3, in <module>\n    something()\n  File "{}", line 2, in something\n    return airflow_DAG\nNameError: name \'airflow_DAG\' is not defined\n'
            assert import_error.stacktrace == expected_stacktrace.format(invalid_dag_filename, invalid_dag_filename)
            session.rollback()

    @conf_vars({('core', 'dagbag_import_error_traceback_depth'): '1'})
    def test_import_error_tracebacks_zip_depth(self, tmpdir):
        if False:
            i = 10
            return i + 15
        invalid_zip_filename = os.path.join(tmpdir, 'test_zip_invalid.zip')
        invalid_dag_filename = os.path.join(invalid_zip_filename, TEMP_DAG_FILENAME)
        with ZipFile(invalid_zip_filename, 'w') as invalid_zip_file:
            invalid_zip_file.writestr(TEMP_DAG_FILENAME, INVALID_DAG_WITH_DEPTH_FILE_CONTENTS)
        with create_session() as session:
            self._process_file(invalid_zip_filename, dag_directory=tmpdir, session=session)
            import_errors = session.query(errors.ImportError).all()
            assert len(import_errors) == 1
            import_error = import_errors[0]
            assert import_error.filename == invalid_dag_filename
            if PY311:
                expected_stacktrace = 'Traceback (most recent call last):\n  File "{}", line 2, in something\n    return airflow_DAG\n           ^^^^^^^^^^^\nNameError: name \'airflow_DAG\' is not defined\n'
            else:
                expected_stacktrace = 'Traceback (most recent call last):\n  File "{}", line 2, in something\n    return airflow_DAG\nNameError: name \'airflow_DAG\' is not defined\n'
            assert import_error.stacktrace == expected_stacktrace.format(invalid_dag_filename)
            session.rollback()

    @conf_vars({('logging', 'dag_processor_log_target'): 'stdout'})
    @mock.patch('airflow.dag_processing.processor.settings.dispose_orm', MagicMock)
    @mock.patch('airflow.dag_processing.processor.redirect_stdout')
    def test_dag_parser_output_when_logging_to_stdout(self, mock_redirect_stdout_for_file):
        if False:
            i = 10
            return i + 15
        processor = DagFileProcessorProcess(file_path='abc.txt', pickle_dags=False, dag_ids=[], dag_directory=[], callback_requests=[])
        processor._run_file_processor(result_channel=MagicMock(), parent_channel=MagicMock(), file_path='fake_file_path', pickle_dags=False, dag_ids=[], thread_name='fake_thread_name', callback_requests=[], dag_directory=[])
        mock_redirect_stdout_for_file.assert_not_called()

    @conf_vars({('logging', 'dag_processor_log_target'): 'file'})
    @mock.patch('airflow.dag_processing.processor.settings.dispose_orm', MagicMock)
    @mock.patch('airflow.dag_processing.processor.redirect_stdout')
    def test_dag_parser_output_when_logging_to_file(self, mock_redirect_stdout_for_file):
        if False:
            for i in range(10):
                print('nop')
        processor = DagFileProcessorProcess(file_path='abc.txt', pickle_dags=False, dag_ids=[], dag_directory=[], callback_requests=[])
        processor._run_file_processor(result_channel=MagicMock(), parent_channel=MagicMock(), file_path='fake_file_path', pickle_dags=False, dag_ids=[], thread_name='fake_thread_name', callback_requests=[], dag_directory=[])
        mock_redirect_stdout_for_file.assert_called_once()

    @mock.patch('airflow.dag_processing.processor.settings.dispose_orm', MagicMock)
    @mock.patch.object(DagFileProcessorProcess, '_get_multiprocessing_context')
    def test_no_valueerror_with_parseable_dag_in_zip(self, mock_context, tmpdir):
        if False:
            while True:
                i = 10
        mock_context.return_value.Pipe.return_value = (MagicMock(), MagicMock())
        zip_filename = os.path.join(tmpdir, 'test_zip.zip')
        with ZipFile(zip_filename, 'w') as zip_file:
            zip_file.writestr(TEMP_DAG_FILENAME, PARSEABLE_DAG_FILE_CONTENTS)
        processor = DagFileProcessorProcess(file_path=zip_filename, pickle_dags=False, dag_ids=[], dag_directory=[], callback_requests=[])
        processor.start()

    @mock.patch('airflow.dag_processing.processor.settings.dispose_orm', MagicMock)
    @mock.patch.object(DagFileProcessorProcess, '_get_multiprocessing_context')
    def test_nullbyte_exception_handling_when_preimporting_airflow(self, mock_context, tmpdir):
        if False:
            i = 10
            return i + 15
        mock_context.return_value.Pipe.return_value = (MagicMock(), MagicMock())
        dag_filename = os.path.join(tmpdir, 'test_dag.py')
        with open(dag_filename, 'wb') as file:
            file.write(b'hello\x00world')
        processor = DagFileProcessorProcess(file_path=dag_filename, pickle_dags=False, dag_ids=[], dag_directory=[], callback_requests=[])
        processor.start()

class TestProcessorAgent:

    @pytest.fixture(autouse=True)
    def per_test(self):
        if False:
            for i in range(10):
                print('nop')
        self.processor_agent = None
        yield
        if self.processor_agent:
            self.processor_agent.end()

    def test_error_when_waiting_in_async_mode(self, tmp_path):
        if False:
            while True:
                i = 10
        self.processor_agent = DagFileProcessorAgent(dag_directory=tmp_path, max_runs=1, processor_timeout=datetime.timedelta(1), dag_ids=[], pickle_dags=False, async_mode=True)
        self.processor_agent.start()
        with pytest.raises(RuntimeError, match='wait_until_finished should only be called in sync_mode'):
            self.processor_agent.wait_until_finished()

    def test_default_multiprocessing_behaviour(self, tmp_path):
        if False:
            for i in range(10):
                print('nop')
        self.processor_agent = DagFileProcessorAgent(dag_directory=tmp_path, max_runs=1, processor_timeout=datetime.timedelta(1), dag_ids=[], pickle_dags=False, async_mode=False)
        self.processor_agent.start()
        self.processor_agent.run_single_parsing_loop()
        self.processor_agent.wait_until_finished()

    @conf_vars({('core', 'mp_start_method'): 'spawn'})
    def test_spawn_multiprocessing_behaviour(self, tmp_path):
        if False:
            for i in range(10):
                print('nop')
        self.processor_agent = DagFileProcessorAgent(dag_directory=tmp_path, max_runs=1, processor_timeout=datetime.timedelta(1), dag_ids=[], pickle_dags=False, async_mode=False)
        self.processor_agent.start()
        self.processor_agent.run_single_parsing_loop()
        self.processor_agent.wait_until_finished()