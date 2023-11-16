from __future__ import annotations
from unittest.mock import Mock, patch
import pytest
from airflow.exceptions import AirflowException
from airflow.providers.tableau.hooks.tableau import TableauJobFinishCode
from airflow.providers.tableau.operators.tableau import TableauOperator

class TestTableauOperator:
    """
    Test class for TableauOperator
    """

    def setup_method(self):
        if False:
            return 10
        self.mocked_workbooks = []
        self.mock_datasources = []
        for i in range(3):
            mock_workbook = Mock()
            mock_workbook.id = i
            mock_workbook.name = f'wb_{i}'
            self.mocked_workbooks.append(mock_workbook)
            mock_datasource = Mock()
            mock_datasource.id = i
            mock_datasource.name = f'ds_{i}'
            self.mock_datasources.append(mock_datasource)
        self.kwargs = {'site_id': 'test_site', 'task_id': 'task', 'dag': None, 'match_with': 'name', 'method': 'refresh'}

    @patch('airflow.providers.tableau.operators.tableau.TableauHook')
    def test_execute_workbooks(self, mock_tableau_hook):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test Execute Workbooks\n        '
        mock_tableau_hook.get_all = Mock(return_value=self.mocked_workbooks)
        mock_tableau_hook.return_value.__enter__ = Mock(return_value=mock_tableau_hook)
        operator = TableauOperator(blocking_refresh=False, find='wb_2', resource='workbooks', **self.kwargs)
        job_id = operator.execute(context={})
        mock_tableau_hook.server.workbooks.refresh.assert_called_once_with(2)
        assert mock_tableau_hook.server.workbooks.refresh.return_value.id == job_id

    @patch('airflow.providers.tableau.operators.tableau.TableauHook')
    def test_execute_workbooks_blocking(self, mock_tableau_hook):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test execute workbooks blocking\n        '
        mock_signed_in = [False]

        def mock_hook_enter():
            if False:
                while True:
                    i = 10
            mock_signed_in[0] = True
            return mock_tableau_hook

        def mock_hook_exit(exc_type, exc_val, exc_tb):
            if False:
                print('Hello World!')
            mock_signed_in[0] = False

        def mock_wait_for_state(job_id, target_state, check_interval):
            if False:
                i = 10
                return i + 15
            if not mock_signed_in[0]:
                raise Exception('Not signed in')
            return True
        mock_tableau_hook.return_value.__enter__ = Mock(side_effect=mock_hook_enter)
        mock_tableau_hook.return_value.__exit__ = Mock(side_effect=mock_hook_exit)
        mock_tableau_hook.wait_for_state = Mock(side_effect=mock_wait_for_state)
        mock_tableau_hook.get_all = Mock(return_value=self.mocked_workbooks)
        mock_tableau_hook.server.jobs.get_by_id = Mock(return_value=Mock(finish_code=TableauJobFinishCode.SUCCESS.value))
        operator = TableauOperator(find='wb_2', resource='workbooks', **self.kwargs)
        job_id = operator.execute(context={})
        mock_tableau_hook.server.workbooks.refresh.assert_called_once_with(2)
        assert mock_tableau_hook.server.workbooks.refresh.return_value.id == job_id
        mock_tableau_hook.wait_for_state.assert_called_once_with(job_id=job_id, check_interval=20, target_state=TableauJobFinishCode.SUCCESS)

    @patch('airflow.providers.tableau.operators.tableau.TableauHook')
    def test_execute_missing_workbook(self, mock_tableau_hook):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test execute missing workbook\n        '
        mock_tableau_hook.get_all = Mock(return_value=self.mocked_workbooks)
        mock_tableau_hook.return_value.__enter__ = Mock(return_value=mock_tableau_hook)
        operator = TableauOperator(find='test', resource='workbooks', **self.kwargs)
        with pytest.raises(AirflowException):
            operator.execute({})

    @patch('airflow.providers.tableau.operators.tableau.TableauHook')
    def test_execute_datasources(self, mock_tableau_hook):
        if False:
            i = 10
            return i + 15
        '\n        Test Execute datasources\n        '
        mock_tableau_hook.get_all = Mock(return_value=self.mock_datasources)
        mock_tableau_hook.return_value.__enter__ = Mock(return_value=mock_tableau_hook)
        operator = TableauOperator(blocking_refresh=False, find='ds_2', resource='datasources', **self.kwargs)
        job_id = operator.execute(context={})
        mock_tableau_hook.server.datasources.refresh.assert_called_once_with(2)
        assert mock_tableau_hook.server.datasources.refresh.return_value.id == job_id

    @patch('airflow.providers.tableau.operators.tableau.TableauHook')
    def test_execute_datasources_blocking(self, mock_tableau_hook):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test execute datasources blocking\n        '
        mock_signed_in = [False]

        def mock_hook_enter():
            if False:
                i = 10
                return i + 15
            mock_signed_in[0] = True
            return mock_tableau_hook

        def mock_hook_exit(exc_type, exc_val, exc_tb):
            if False:
                while True:
                    i = 10
            mock_signed_in[0] = False

        def mock_wait_for_state(job_id, target_state, check_interval):
            if False:
                while True:
                    i = 10
            if not mock_signed_in[0]:
                raise Exception('Not signed in')
            return True
        mock_tableau_hook.return_value.__enter__ = Mock(side_effect=mock_hook_enter)
        mock_tableau_hook.return_value.__exit__ = Mock(side_effect=mock_hook_exit)
        mock_tableau_hook.wait_for_state = Mock(side_effect=mock_wait_for_state)
        mock_tableau_hook.get_all = Mock(return_value=self.mock_datasources)
        operator = TableauOperator(find='ds_2', resource='datasources', **self.kwargs)
        job_id = operator.execute(context={})
        mock_tableau_hook.server.datasources.refresh.assert_called_once_with(2)
        assert mock_tableau_hook.server.datasources.refresh.return_value.id == job_id
        mock_tableau_hook.wait_for_state.assert_called_once_with(job_id=job_id, check_interval=20, target_state=TableauJobFinishCode.SUCCESS)

    @patch('airflow.providers.tableau.operators.tableau.TableauHook')
    def test_execute_missing_datasource(self, mock_tableau_hook):
        if False:
            print('Hello World!')
        '\n        Test execute missing datasource\n        '
        mock_tableau_hook.get_all = Mock(return_value=self.mock_datasources)
        mock_tableau_hook.return_value.__enter__ = Mock(return_value=mock_tableau_hook)
        operator = TableauOperator(find='test', resource='datasources', **self.kwargs)
        with pytest.raises(AirflowException):
            operator.execute({})

    def test_execute_unavailable_resource(self):
        if False:
            i = 10
            return i + 15
        '\n        Test execute unavailable resource\n        '
        operator = TableauOperator(resource='test', find='test', **self.kwargs)
        with pytest.raises(AirflowException):
            operator.execute({})

    def test_get_resource_id(self):
        if False:
            i = 10
            return i + 15
        '\n        Test get resource id\n        '
        resource_id = 'res_id'
        operator = TableauOperator(resource='task', find=resource_id, method='run', task_id='t', dag=None)
        assert operator._get_resource_id(resource_id) == resource_id