from __future__ import annotations
import pytest
from airflow.decorators import task
from airflow.utils.state import State
pytestmark = pytest.mark.db_test

class Test_BranchPythonVirtualenvDecoratedOperator:

    @pytest.mark.execution_timeout(180)
    @pytest.mark.parametrize('branch_task_name', ['task_1', 'task_2'])
    def test_branch_one(self, dag_maker, branch_task_name):
        if False:
            print('Hello World!')

        @task
        def dummy_f():
            if False:
                for i in range(10):
                    print('nop')
            pass

        @task
        def task_1():
            if False:
                while True:
                    i = 10
            pass

        @task
        def task_2():
            if False:
                return 10
            pass
        if branch_task_name == 'task_1':

            @task.branch_virtualenv(task_id='branching', requirements=['funcsigs'])
            def branch_operator():
                if False:
                    i = 10
                    return i + 15
                import funcsigs
                print(f'We successfully imported funcsigs version {funcsigs.__version__}')
                return 'task_1'
        else:

            @task.branch_virtualenv(task_id='branching', requirements=['funcsigs'])
            def branch_operator():
                if False:
                    return 10
                import funcsigs
                print(f'We successfully imported funcsigs version {funcsigs.__version__}')
                return 'task_2'
        with dag_maker():
            branchoperator = branch_operator()
            df = dummy_f()
            task_1 = task_1()
            task_2 = task_2()
            df.set_downstream(branchoperator)
            branchoperator.set_downstream(task_1)
            branchoperator.set_downstream(task_2)
        dr = dag_maker.create_dagrun()
        df.operator.run(start_date=dr.execution_date, end_date=dr.execution_date, ignore_ti_state=True)
        branchoperator.operator.run(start_date=dr.execution_date, end_date=dr.execution_date, ignore_ti_state=True)
        task_1.operator.run(start_date=dr.execution_date, end_date=dr.execution_date, ignore_ti_state=True)
        task_2.operator.run(start_date=dr.execution_date, end_date=dr.execution_date, ignore_ti_state=True)
        tis = dr.get_task_instances()
        for ti in tis:
            if ti.task_id == 'dummy_f':
                assert ti.state == State.SUCCESS
            if ti.task_id == 'branching':
                assert ti.state == State.SUCCESS
            if ti.task_id == 'task_1' and branch_task_name == 'task_1':
                assert ti.state == State.SUCCESS
            elif ti.task_id == 'task_1':
                assert ti.state == State.SKIPPED
            if ti.task_id == 'task_2' and branch_task_name == 'task_2':
                assert ti.state == State.SUCCESS
            elif ti.task_id == 'task_2':
                assert ti.state == State.SKIPPED