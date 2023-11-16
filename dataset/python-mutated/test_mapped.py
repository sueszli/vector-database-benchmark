from __future__ import annotations
from airflow.models.dag import DAG
from airflow.utils.task_group import TaskGroup
from tests.models import DEFAULT_DATE

def test_mapped_task_group_id_prefix_task_id():
    if False:
        i = 10
        return i + 15

    def f(z):
        if False:
            print('Hello World!')
        pass
    with DAG(dag_id='d', start_date=DEFAULT_DATE) as dag:
        x1 = dag.task(task_id='t1')(f).expand(z=[])
        with TaskGroup('g'):
            x2 = dag.task(task_id='t2')(f).expand(z=[])
    assert x1.operator.task_id == 't1'
    assert x2.operator.task_id == 'g.t2'
    dag.get_task('t1') == x1.operator
    dag.get_task('g.t2') == x2.operator