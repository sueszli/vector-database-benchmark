from __future__ import annotations
from unittest import mock
from unittest.mock import MagicMock
import pytest
from airflow.exceptions import AirflowException
from airflow.models import DAG, DagBag
from airflow.providers.google.cloud.operators.looker import LookerStartPdtBuildOperator
from airflow.utils.timezone import datetime
from tests.test_utils.db import clear_db_runs, clear_db_xcom
OPERATOR_PATH = 'airflow.providers.google.cloud.operators.looker.{}'
TASK_ID = 'task-id'
LOOKER_CONN_ID = 'test-conn'
MODEL = 'test_model'
VIEW = 'test_view'
TEST_DAG_ID = 'test-looker-operators'
DEFAULT_DATE = datetime(2020, 1, 1)
TEST_JOB_ID = '123'

class LookerTestBase:

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        cls.dagbag = DagBag(dag_folder='/dev/null', include_examples=False)
        cls.dag = DAG(TEST_DAG_ID, default_args={'owner': 'airflow', 'start_date': DEFAULT_DATE})

    def setup_method(self):
        if False:
            return 10
        self.mock_ti = MagicMock()
        self.mock_context = {'ti': self.mock_ti}

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.mock_ti = MagicMock()
        self.mock_context = {'ti': self.mock_ti}

    @classmethod
    def tearDownClass(cls):
        if False:
            return 10
        clear_db_runs()
        clear_db_xcom()

class TestLookerStartPdtBuildOperator(LookerTestBase):

    @mock.patch(OPERATOR_PATH.format('LookerHook'))
    def test_execute(self, mock_hook):
        if False:
            print('Hello World!')
        mock_hook.return_value.start_pdt_build.return_value.materialization_id = TEST_JOB_ID
        mock_hook.return_value.wait_for_job.return_value = None
        task = LookerStartPdtBuildOperator(task_id=TASK_ID, looker_conn_id=LOOKER_CONN_ID, model=MODEL, view=VIEW)
        task.execute(context=self.mock_context)
        mock_hook.assert_called_once_with(looker_conn_id=LOOKER_CONN_ID)
        mock_hook.return_value.start_pdt_build.assert_called_once_with(model=MODEL, view=VIEW, query_params=None)
        mock_hook.return_value.wait_for_job.assert_called_once_with(materialization_id=TEST_JOB_ID, wait_time=10, timeout=None)

    @mock.patch(OPERATOR_PATH.format('LookerHook'))
    def test_execute_async(self, mock_hook):
        if False:
            while True:
                i = 10
        mock_hook.return_value.start_pdt_build.return_value.materialization_id = TEST_JOB_ID
        mock_hook.return_value.wait_for_job.return_value = None
        task = LookerStartPdtBuildOperator(task_id=TASK_ID, looker_conn_id=LOOKER_CONN_ID, model=MODEL, view=VIEW, asynchronous=True)
        task.execute(context=self.mock_context)
        mock_hook.assert_called_once_with(looker_conn_id=LOOKER_CONN_ID)
        mock_hook.return_value.start_pdt_build.assert_called_once_with(model=MODEL, view=VIEW, query_params=None)
        mock_hook.return_value.wait_for_job.assert_not_called()

    @mock.patch(OPERATOR_PATH.format('LookerHook'))
    def test_on_kill(self, mock_hook):
        if False:
            for i in range(10):
                print('nop')
        mock_hook.return_value.start_pdt_build.return_value.materialization_id = TEST_JOB_ID
        mock_hook.return_value.wait_for_job.return_value = None
        task = LookerStartPdtBuildOperator(task_id=TASK_ID, looker_conn_id=LOOKER_CONN_ID, model=MODEL, view=VIEW, cancel_on_kill=False)
        task.execute(context=self.mock_context)
        task.on_kill()
        mock_hook.return_value.stop_pdt_build.assert_not_called()
        task.cancel_on_kill = True
        task.on_kill()
        mock_hook.return_value.stop_pdt_build.assert_called_once_with(materialization_id=TEST_JOB_ID)

    @mock.patch(OPERATOR_PATH.format('LookerHook'))
    def test_materialization_id_returned_as_empty_str(self, mock_hook):
        if False:
            return 10
        mock_hook.return_value.start_pdt_build.return_value.materialization_id = ''
        mock_hook.return_value.wait_for_job.return_value = None
        task = LookerStartPdtBuildOperator(task_id=TASK_ID, looker_conn_id=LOOKER_CONN_ID, model=MODEL, view=VIEW)
        with pytest.raises(AirflowException, match=f'No `materialization_id` was returned for model: {MODEL}, view: {VIEW}.'):
            task.execute(context=self.mock_context)