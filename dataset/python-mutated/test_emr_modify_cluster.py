from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest
from airflow.exceptions import AirflowException
from airflow.models.dag import DAG
from airflow.providers.amazon.aws.operators.emr import EmrModifyClusterOperator
from airflow.utils import timezone
DEFAULT_DATE = timezone.datetime(2017, 1, 1)
MODIFY_CLUSTER_SUCCESS_RETURN = {'ResponseMetadata': {'HTTPStatusCode': 200}, 'StepConcurrencyLevel': 1}
MODIFY_CLUSTER_ERROR_RETURN = {'ResponseMetadata': {'HTTPStatusCode': 400}}

@pytest.fixture
def mocked_hook_client():
    if False:
        return 10
    with patch('airflow.providers.amazon.aws.hooks.emr.EmrHook.conn') as m:
        yield m

class TestEmrModifyClusterOperator:

    def setup_method(self):
        if False:
            for i in range(10):
                print('nop')
        args = {'owner': 'airflow', 'start_date': DEFAULT_DATE}
        self.mock_context = MagicMock()
        self.operator = EmrModifyClusterOperator(task_id='test_task', cluster_id='j-8989898989', step_concurrency_level=1, aws_conn_id='aws_default', dag=DAG('test_dag_id', default_args=args))

    def test_init(self):
        if False:
            i = 10
            return i + 15
        assert self.operator.cluster_id == 'j-8989898989'
        assert self.operator.step_concurrency_level == 1
        assert self.operator.aws_conn_id == 'aws_default'

    def test_execute_returns_step_concurrency(self, mocked_hook_client):
        if False:
            while True:
                i = 10
        mocked_hook_client.modify_cluster.return_value = MODIFY_CLUSTER_SUCCESS_RETURN
        assert self.operator.execute(self.mock_context) == 1

    def test_execute_returns_error(self, mocked_hook_client):
        if False:
            i = 10
            return i + 15
        mocked_hook_client.modify_cluster.return_value = MODIFY_CLUSTER_ERROR_RETURN
        with pytest.raises(AirflowException, match='Modify cluster failed'):
            self.operator.execute(self.mock_context)