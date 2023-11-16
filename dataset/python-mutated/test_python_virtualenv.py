from __future__ import annotations
import datetime
import sys
from subprocess import CalledProcessError
import pytest
from airflow.decorators import setup, task, teardown
from airflow.utils import timezone
pytestmark = pytest.mark.db_test
DEFAULT_DATE = timezone.datetime(2016, 1, 1)
PYTHON_VERSION = sys.version_info[0]

class TestPythonVirtualenvDecorator:

    def test_add_dill(self, dag_maker):
        if False:
            i = 10
            return i + 15

        @task.virtualenv(use_dill=True, system_site_packages=False)
        def f():
            if False:
                for i in range(10):
                    print('nop')
            'Ensure dill is correctly installed.'
            import dill
        with dag_maker():
            ret = f()
        ret.operator.run(start_date=DEFAULT_DATE, end_date=DEFAULT_DATE)

    def test_no_requirements(self, dag_maker):
        if False:
            return 10
        'Tests that the python callable is invoked on task run.'

        @task.virtualenv()
        def f():
            if False:
                return 10
            pass
        with dag_maker():
            ret = f()
        ret.operator.run(start_date=DEFAULT_DATE, end_date=DEFAULT_DATE)

    def test_no_system_site_packages(self, dag_maker):
        if False:
            for i in range(10):
                print('nop')

        @task.virtualenv(system_site_packages=False, python_version=PYTHON_VERSION, use_dill=True)
        def f():
            if False:
                for i in range(10):
                    print('nop')
            try:
                import funcsigs
            except ImportError:
                return True
            raise Exception
        with dag_maker():
            ret = f()
        ret.operator.run(start_date=DEFAULT_DATE, end_date=DEFAULT_DATE)

    def test_system_site_packages(self, dag_maker):
        if False:
            while True:
                i = 10

        @task.virtualenv(system_site_packages=False, requirements=['funcsigs'], python_version=PYTHON_VERSION, use_dill=True)
        def f():
            if False:
                print('Hello World!')
            import funcsigs
        with dag_maker():
            ret = f()
        ret.operator.run(start_date=DEFAULT_DATE, end_date=DEFAULT_DATE)

    def test_with_requirements_pinned(self, dag_maker):
        if False:
            for i in range(10):
                print('nop')

        @task.virtualenv(system_site_packages=False, requirements=['funcsigs==0.4'], python_version=PYTHON_VERSION, use_dill=True)
        def f():
            if False:
                return 10
            import funcsigs
            if funcsigs.__version__ != '0.4':
                raise Exception
        with dag_maker():
            ret = f()
        ret.operator.run(start_date=DEFAULT_DATE, end_date=DEFAULT_DATE)

    def test_unpinned_requirements(self, dag_maker):
        if False:
            i = 10
            return i + 15

        @task.virtualenv(system_site_packages=False, requirements=['funcsigs', 'dill'], python_version=PYTHON_VERSION, use_dill=True)
        def f():
            if False:
                print('Hello World!')
            import funcsigs
        with dag_maker():
            ret = f()
        ret.operator.run(start_date=DEFAULT_DATE, end_date=DEFAULT_DATE)

    def test_fail(self, dag_maker):
        if False:
            print('Hello World!')

        @task.virtualenv()
        def f():
            if False:
                for i in range(10):
                    print('nop')
            raise Exception
        with dag_maker():
            ret = f()
        with pytest.raises(CalledProcessError):
            ret.operator.run(start_date=DEFAULT_DATE, end_date=DEFAULT_DATE)

    def test_python_3(self, dag_maker):
        if False:
            i = 10
            return i + 15

        @task.virtualenv(python_version=3, use_dill=False, requirements=['dill'])
        def f():
            if False:
                for i in range(10):
                    print('nop')
            import sys
            print(sys.version)
            try:
                {}.iteritems()
            except AttributeError:
                return
            raise Exception
        with dag_maker():
            ret = f()
        ret.operator.run(start_date=DEFAULT_DATE, end_date=DEFAULT_DATE)

    def test_with_args(self, dag_maker):
        if False:
            i = 10
            return i + 15

        @task.virtualenv
        def f(a, b, c=False, d=False):
            if False:
                for i in range(10):
                    print('nop')
            if a == 0 and b == 1 and c and (not d):
                return True
            else:
                raise Exception
        with dag_maker():
            ret = f(0, 1, c=True)
        ret.operator.run(start_date=DEFAULT_DATE, end_date=DEFAULT_DATE)

    def test_return_none(self, dag_maker):
        if False:
            i = 10
            return i + 15

        @task.virtualenv
        def f():
            if False:
                i = 10
                return i + 15
            return None
        with dag_maker():
            ret = f()
        ret.operator.run(start_date=DEFAULT_DATE, end_date=DEFAULT_DATE)

    def test_nonimported_as_arg(self, dag_maker):
        if False:
            return 10

        @task.virtualenv
        def f(_):
            if False:
                for i in range(10):
                    print('nop')
            return None
        with dag_maker():
            ret = f(datetime.datetime.utcnow())
        ret.operator.run(start_date=DEFAULT_DATE, end_date=DEFAULT_DATE)

    def test_marking_virtualenv_python_task_as_setup(self, dag_maker):
        if False:
            while True:
                i = 10

        @setup
        @task.virtualenv
        def f():
            if False:
                i = 10
                return i + 15
            return 1
        with dag_maker() as dag:
            ret = f()
        assert len(dag.task_group.children) == 1
        setup_task = dag.task_group.children['f']
        assert setup_task.is_setup
        ret.operator.run(start_date=DEFAULT_DATE, end_date=DEFAULT_DATE)

    def test_marking_virtualenv_python_task_as_teardown(self, dag_maker):
        if False:
            for i in range(10):
                print('nop')

        @teardown
        @task.virtualenv
        def f():
            if False:
                i = 10
                return i + 15
            return 1
        with dag_maker() as dag:
            ret = f()
        assert len(dag.task_group.children) == 1
        teardown_task = dag.task_group.children['f']
        assert teardown_task.is_teardown
        ret.operator.run(start_date=DEFAULT_DATE, end_date=DEFAULT_DATE)

    @pytest.mark.parametrize('on_failure_fail_dagrun', [True, False])
    def test_marking_virtualenv_python_task_as_teardown_with_on_failure_fail(self, dag_maker, on_failure_fail_dagrun):
        if False:
            return 10

        @teardown(on_failure_fail_dagrun=on_failure_fail_dagrun)
        @task.virtualenv
        def f():
            if False:
                for i in range(10):
                    print('nop')
            return 1
        with dag_maker() as dag:
            ret = f()
        assert len(dag.task_group.children) == 1
        teardown_task = dag.task_group.children['f']
        assert teardown_task.is_teardown
        assert teardown_task.on_failure_fail_dagrun is on_failure_fail_dagrun
        ret.operator.run(start_date=DEFAULT_DATE, end_date=DEFAULT_DATE)