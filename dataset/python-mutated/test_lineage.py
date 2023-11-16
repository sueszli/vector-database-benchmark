from __future__ import annotations
from unittest import mock
import attr
import pytest
from airflow.lineage import AUTO, apply_lineage, get_backend, prepare_lineage
from airflow.lineage.backend import LineageBackend
from airflow.lineage.entities import File
from airflow.models import TaskInstance as TI
from airflow.operators.empty import EmptyOperator
from airflow.utils import timezone
from airflow.utils.context import Context
from airflow.utils.types import DagRunType
from tests.test_utils.config import conf_vars
pytestmark = pytest.mark.db_test
DEFAULT_DATE = timezone.datetime(2016, 1, 1)

@attr.define
class A:
    pass

class CustomLineageBackend(LineageBackend):

    def send_lineage(self, operator, inlets=None, outlets=None, context=None):
        if False:
            while True:
                i = 10
        pass

class TestLineage:

    def test_lineage(self, dag_maker):
        if False:
            while True:
                i = 10
        f1s = '/tmp/does_not_exist_1-{}'
        f2s = '/tmp/does_not_exist_2-{}'
        f3s = '/tmp/does_not_exist_3'
        file1 = File(f1s.format('{{ ds }}'))
        file2 = File(f2s.format('{{ ds }}'))
        file3 = File(f3s)
        with dag_maker(dag_id='test_prepare_lineage', start_date=DEFAULT_DATE) as dag:
            op1 = EmptyOperator(task_id='leave1', inlets=file1, outlets=[file2])
            op2 = EmptyOperator(task_id='leave2')
            op3 = EmptyOperator(task_id='upstream_level_1', inlets=AUTO, outlets=file3)
            op4 = EmptyOperator(task_id='upstream_level_2')
            op5 = EmptyOperator(task_id='upstream_level_3', inlets=['leave1', 'upstream_level_1'])
            op1.set_downstream(op3)
            op2.set_downstream(op3)
            op3.set_downstream(op4)
            op4.set_downstream(op5)
        dag.clear()
        dag_run = dag_maker.create_dagrun(run_type=DagRunType.SCHEDULED)
        ctx1 = Context({'ti': TI(task=op1, run_id=dag_run.run_id), 'ds': DEFAULT_DATE})
        ctx2 = Context({'ti': TI(task=op2, run_id=dag_run.run_id), 'ds': DEFAULT_DATE})
        ctx3 = Context({'ti': TI(task=op3, run_id=dag_run.run_id), 'ds': DEFAULT_DATE})
        ctx5 = Context({'ti': TI(task=op5, run_id=dag_run.run_id), 'ds': DEFAULT_DATE})
        op1.pre_execute(ctx1)
        assert len(op1.inlets) == 1
        assert op1.inlets[0].url == f1s.format(DEFAULT_DATE)
        assert len(op1.outlets) == 1
        assert op1.outlets[0].url == f2s.format(DEFAULT_DATE)
        op1.post_execute(ctx1)
        op2.pre_execute(ctx2)
        assert len(op2.inlets) == 0
        op2.post_execute(ctx2)
        op3.pre_execute(ctx3)
        assert len(op3.inlets) == 1
        assert isinstance(op3.inlets[0], File)
        assert op3.inlets[0].url == f2s.format(DEFAULT_DATE)
        assert op3.outlets[0] == file3
        op3.post_execute(ctx3)
        op5.pre_execute(ctx5)
        assert sorted(op5.inlets) == [file2, file3]
        op5.post_execute(ctx5)

    def test_lineage_render(self, dag_maker):
        if False:
            while True:
                i = 10
        with dag_maker(dag_id='test_lineage_render', start_date=DEFAULT_DATE):
            op1 = EmptyOperator(task_id='task1')
        dag_run = dag_maker.create_dagrun(run_type=DagRunType.SCHEDULED)
        f1s = '/tmp/does_not_exist_1-{}'
        file1 = File(f1s.format('{{ ds }}'))
        op1.inlets.append(file1)
        op1.outlets.append(file1)
        ctx1 = Context({'ti': TI(task=op1, run_id=dag_run.run_id), 'ds': DEFAULT_DATE})
        op1.pre_execute(ctx1)
        assert op1.inlets[0].url == f1s.format(DEFAULT_DATE)
        assert op1.outlets[0].url == f1s.format(DEFAULT_DATE)

    def test_attr_outlet(self, dag_maker):
        if False:
            while True:
                i = 10
        a = A()
        f3s = '/tmp/does_not_exist_3'
        file3 = File(f3s)
        with dag_maker(dag_id='test_prepare_lineage'):
            op1 = EmptyOperator(task_id='leave1', outlets=[a, file3])
            op2 = EmptyOperator(task_id='leave2', inlets='auto')
            op1 >> op2
        dag_run = dag_maker.create_dagrun(run_type=DagRunType.SCHEDULED)
        ctx1 = Context({'ti': TI(task=op1, run_id=dag_run.run_id), 'ds': DEFAULT_DATE})
        ctx2 = Context({'ti': TI(task=op2, run_id=dag_run.run_id), 'ds': DEFAULT_DATE})
        op1.pre_execute(ctx1)
        op1.post_execute(ctx1)
        op2.pre_execute(ctx2)
        assert op2.inlets == [a, file3]
        op2.post_execute(ctx2)

    @mock.patch('airflow.lineage.get_backend')
    def test_lineage_is_sent_to_backend(self, mock_get_backend, dag_maker):
        if False:
            while True:
                i = 10

        class TestBackend(LineageBackend):

            def send_lineage(self, operator, inlets=None, outlets=None, context=None):
                if False:
                    for i in range(10):
                        print('nop')
                assert len(inlets) == 1
                assert len(outlets) == 1
        func = mock.Mock()
        func.__name__ = 'foo'
        mock_get_backend.return_value = TestBackend()
        with dag_maker(dag_id='test_lineage_is_sent_to_backend', start_date=DEFAULT_DATE):
            op1 = EmptyOperator(task_id='task1')
        dag_run = dag_maker.create_dagrun(run_type=DagRunType.SCHEDULED)
        file1 = File('/tmp/some_file')
        op1.inlets.append(file1)
        op1.outlets.append(file1)
        (ti,) = dag_run.task_instances
        ctx1 = Context({'ti': ti, 'ds': DEFAULT_DATE})
        prep = prepare_lineage(func)
        prep(op1, ctx1)
        post = apply_lineage(func)
        post(op1, ctx1)

    def test_empty_lineage_backend(self):
        if False:
            for i in range(10):
                print('nop')
        backend = get_backend()
        assert backend is None

    @conf_vars({('lineage', 'backend'): 'tests.lineage.test_lineage.CustomLineageBackend'})
    def test_resolve_lineage_class(self):
        if False:
            for i in range(10):
                print('nop')
        backend = get_backend()
        assert issubclass(backend.__class__, LineageBackend)
        assert isinstance(backend, CustomLineageBackend)