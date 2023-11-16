from __future__ import annotations
import pytest
from airflow.exceptions import AirflowDagCycleException
from airflow.models.dag import DAG
from airflow.operators.empty import EmptyOperator
from airflow.utils.dag_cycle_tester import check_cycle
from airflow.utils.edgemodifier import Label
from airflow.utils.task_group import TaskGroup
from tests.models import DEFAULT_DATE

class TestCycleTester:

    def test_cycle_empty(self):
        if False:
            return 10
        dag = DAG('dag', start_date=DEFAULT_DATE, default_args={'owner': 'owner1'})
        assert not check_cycle(dag)

    def test_cycle_single_task(self):
        if False:
            for i in range(10):
                print('nop')
        dag = DAG('dag', start_date=DEFAULT_DATE, default_args={'owner': 'owner1'})
        with dag:
            EmptyOperator(task_id='A')
        assert not check_cycle(dag)

    def test_semi_complex(self):
        if False:
            for i in range(10):
                print('nop')
        dag = DAG('dag', start_date=DEFAULT_DATE, default_args={'owner': 'owner1'})
        with dag:
            create_cluster = EmptyOperator(task_id='c')
            pod_task = EmptyOperator(task_id='p')
            pod_task_xcom = EmptyOperator(task_id='x')
            delete_cluster = EmptyOperator(task_id='d')
            pod_task_xcom_result = EmptyOperator(task_id='r')
            create_cluster >> pod_task >> delete_cluster
            create_cluster >> pod_task_xcom >> delete_cluster
            pod_task_xcom >> pod_task_xcom_result

    def test_cycle_no_cycle(self):
        if False:
            print('Hello World!')
        dag = DAG('dag', start_date=DEFAULT_DATE, default_args={'owner': 'owner1'})
        with dag:
            op1 = EmptyOperator(task_id='A')
            op2 = EmptyOperator(task_id='B')
            op3 = EmptyOperator(task_id='C')
            op4 = EmptyOperator(task_id='D')
            op5 = EmptyOperator(task_id='E')
            op6 = EmptyOperator(task_id='F')
            op1.set_downstream(op2)
            op2.set_downstream(op3)
            op2.set_downstream(op4)
            op5.set_downstream(op6)
        assert not check_cycle(dag)

    def test_cycle_loop(self):
        if False:
            i = 10
            return i + 15
        dag = DAG('dag', start_date=DEFAULT_DATE, default_args={'owner': 'owner1'})
        with dag:
            op1 = EmptyOperator(task_id='A')
            op1.set_downstream(op1)
        with pytest.raises(AirflowDagCycleException):
            assert not check_cycle(dag)

    def test_cycle_downstream_loop(self):
        if False:
            return 10
        dag = DAG('dag', start_date=DEFAULT_DATE, default_args={'owner': 'owner1'})
        with dag:
            op1 = EmptyOperator(task_id='A')
            op2 = EmptyOperator(task_id='B')
            op3 = EmptyOperator(task_id='C')
            op4 = EmptyOperator(task_id='D')
            op5 = EmptyOperator(task_id='E')
            op1.set_downstream(op2)
            op2.set_downstream(op3)
            op3.set_downstream(op4)
            op4.set_downstream(op5)
            op5.set_downstream(op5)
        with pytest.raises(AirflowDagCycleException):
            assert not check_cycle(dag)

    def test_cycle_large_loop(self):
        if False:
            i = 10
            return i + 15
        dag = DAG('dag', start_date=DEFAULT_DATE, default_args={'owner': 'owner1'})
        with dag:
            start = EmptyOperator(task_id='start')
            current = start
            for i in range(10000):
                next_task = EmptyOperator(task_id=f'task_{i}')
                current.set_downstream(next_task)
                current = next_task
            current.set_downstream(start)
        with pytest.raises(AirflowDagCycleException):
            assert not check_cycle(dag)

    def test_cycle_arbitrary_loop(self):
        if False:
            print('Hello World!')
        dag = DAG('dag', start_date=DEFAULT_DATE, default_args={'owner': 'owner1'})
        with dag:
            op1 = EmptyOperator(task_id='A')
            op2 = EmptyOperator(task_id='B')
            op3 = EmptyOperator(task_id='C')
            op4 = EmptyOperator(task_id='E')
            op5 = EmptyOperator(task_id='F')
            op1.set_downstream(op2)
            op1.set_downstream(op3)
            op4.set_downstream(op1)
            op3.set_downstream(op5)
            op2.set_downstream(op5)
            op5.set_downstream(op1)
        with pytest.raises(AirflowDagCycleException):
            assert not check_cycle(dag)

    def test_cycle_task_group_with_edge_labels(self):
        if False:
            i = 10
            return i + 15
        dag = DAG('dag', start_date=DEFAULT_DATE, default_args={'owner': 'owner1'})
        with dag:
            with TaskGroup(group_id='group'):
                op1 = EmptyOperator(task_id='A')
                op2 = EmptyOperator(task_id='B')
                op1 >> Label('label') >> op2
        assert not check_cycle(dag)