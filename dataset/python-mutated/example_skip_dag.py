"""Example DAG demonstrating the EmptyOperator and a custom EmptySkipOperator which skips by default."""
from __future__ import annotations
from typing import TYPE_CHECKING
import pendulum
from airflow.exceptions import AirflowSkipException
from airflow.models.baseoperator import BaseOperator
from airflow.models.dag import DAG
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule
if TYPE_CHECKING:
    from airflow.utils.context import Context

class EmptySkipOperator(BaseOperator):
    """Empty operator which always skips the task."""
    ui_color = '#e8b7e4'

    def execute(self, context: Context):
        if False:
            print('Hello World!')
        raise AirflowSkipException

def create_test_pipeline(suffix, trigger_rule):
    if False:
        return 10
    '\n    Instantiate a number of operators for the given DAG.\n\n    :param str suffix: Suffix to append to the operator task_ids\n    :param str trigger_rule: TriggerRule for the join task\n    :param DAG dag_: The DAG to run the operators on\n    '
    skip_operator = EmptySkipOperator(task_id=f'skip_operator_{suffix}')
    always_true = EmptyOperator(task_id=f'always_true_{suffix}')
    join = EmptyOperator(task_id=trigger_rule, trigger_rule=trigger_rule)
    final = EmptyOperator(task_id=f'final_{suffix}')
    skip_operator >> join
    always_true >> join
    join >> final
with DAG(dag_id='example_skip_dag', start_date=pendulum.datetime(2021, 1, 1, tz='UTC'), catchup=False, tags=['example']) as dag:
    create_test_pipeline('1', TriggerRule.ALL_SUCCESS)
    create_test_pipeline('2', TriggerRule.ONE_SUCCESS)