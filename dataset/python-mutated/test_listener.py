from __future__ import annotations
import datetime
import uuid
from unittest.mock import patch
import pandas as pd
import pytest
from airflow.models import DAG, TaskInstance
from airflow.models.baseoperator import BaseOperator
from airflow.utils.state import State
pytestmark = pytest.mark.db_test

class TemplateOperator(BaseOperator):
    template_fields = ['df']

    def __init__(self, df, *args, **kwargs):
        if False:
            while True:
                i = 10
        self.df = df
        super().__init__(*args, **kwargs)

    def execute(self, context):
        if False:
            i = 10
            return i + 15
        return self.df

def render_df():
    if False:
        i = 10
        return i + 15
    return pd.DataFrame({'col': [1, 2]})

@patch('airflow.models.TaskInstance.xcom_push')
@patch('airflow.models.BaseOperator.render_template')
def test_listener_does_not_change_task_instance(render_mock, xcom_push_mock):
    if False:
        print('Hello World!')
    render_mock.return_value = render_df()
    dag = DAG('test', start_date=datetime.datetime(2022, 1, 1), user_defined_macros={'render_df': render_df}, params={'df': render_df()})
    t = TemplateOperator(task_id='template_op', dag=dag, do_xcom_push=True, df=dag.param('df'))
    run_id = str(uuid.uuid1())
    dag.create_dagrun(state=State.NONE, run_id=run_id)
    ti = TaskInstance(t, run_id=run_id)
    ti.check_and_change_state_before_execution()
    ti._run_raw_task()
    pd.testing.assert_frame_equal(xcom_push_mock.call_args.kwargs['value'], render_df())
    assert not isinstance(render_mock.call_args.args[0], pd.DataFrame)