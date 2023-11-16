from __future__ import annotations
import datetime
from typing import Callable
import pytest
from sqlalchemy.orm import eagerload
from airflow import models
from airflow.api.common.mark_tasks import _create_dagruns, _DagRunInfo, set_dag_run_state_to_failed, set_dag_run_state_to_queued, set_dag_run_state_to_running, set_dag_run_state_to_success, set_state
from airflow.models import DagRun
from airflow.utils import timezone
from airflow.utils.session import create_session, provide_session
from airflow.utils.state import State
from airflow.utils.types import DagRunType
from tests.test_utils.db import clear_db_runs
from tests.test_utils.mapping import expand_mapped_task
DEV_NULL = '/dev/null'
pytestmark = pytest.mark.db_test

@pytest.fixture(scope='module')
def dagbag():
    if False:
        return 10
    from airflow.models.dagbag import DagBag
    non_serialized_dagbag = DagBag(read_dags_from_db=False, include_examples=True)
    non_serialized_dagbag.sync_to_db()
    return DagBag(read_dags_from_db=True)

class TestMarkTasks:

    @pytest.fixture(scope='class', autouse=True, name='create_dags')
    @classmethod
    def create_dags(cls, dagbag):
        if False:
            for i in range(10):
                print('nop')
        cls.dag1 = dagbag.get_dag('miscellaneous_test_dag')
        cls.dag2 = dagbag.get_dag('example_subdag_operator')
        cls.dag3 = dagbag.get_dag('example_trigger_target_dag')
        cls.dag4 = dagbag.get_dag('test_mapped_classic')
        cls.execution_dates = [timezone.datetime(2022, 1, 1), timezone.datetime(2022, 1, 2)]
        start_date3 = cls.dag3.start_date
        cls.dag3_execution_dates = [start_date3, start_date3 + datetime.timedelta(days=1), start_date3 + datetime.timedelta(days=2)]

    @pytest.fixture(autouse=True)
    def setup_tests(self):
        if False:
            return 10
        clear_db_runs()
        drs = _create_dagruns(self.dag1, [_DagRunInfo(d, (d, d + datetime.timedelta(days=1))) for d in self.execution_dates], state=State.RUNNING, run_type=DagRunType.SCHEDULED)
        for dr in drs:
            dr.dag = self.dag1
        drs = _create_dagruns(self.dag2, [_DagRunInfo(self.dag2.start_date, (self.dag2.start_date, self.dag2.start_date + datetime.timedelta(days=1)))], state=State.RUNNING, run_type=DagRunType.SCHEDULED)
        for dr in drs:
            dr.dag = self.dag2
        drs = _create_dagruns(self.dag3, [_DagRunInfo(d, (d, d)) for d in self.dag3_execution_dates], state=State.SUCCESS, run_type=DagRunType.MANUAL)
        for dr in drs:
            dr.dag = self.dag3
        drs = _create_dagruns(self.dag4, [_DagRunInfo(self.dag4.start_date, (self.dag4.start_date, self.dag4.start_date + datetime.timedelta(days=1)))], state=State.SUCCESS, run_type=DagRunType.MANUAL)
        for dr in drs:
            dr.dag = self.dag4
        yield
        clear_db_runs()

    @staticmethod
    def snapshot_state(dag, execution_dates):
        if False:
            for i in range(10):
                print('nop')
        TI = models.TaskInstance
        DR = models.DagRun
        with create_session() as session:
            return session.query(TI).join(TI.dag_run).options(eagerload(TI.dag_run)).filter(TI.dag_id == dag.dag_id, DR.execution_date.in_(execution_dates)).all()

    @provide_session
    def verify_state(self, dag, task_ids, execution_dates, state, old_tis, session=None, map_task_pairs=None):
        if False:
            for i in range(10):
                print('nop')
        TI = models.TaskInstance
        DR = models.DagRun
        tis = session.query(TI).join(TI.dag_run).filter(TI.dag_id == dag.dag_id, DR.execution_date.in_(execution_dates)).all()
        assert len(tis) > 0
        unexpected_tis = []
        for ti in tis:
            assert ti.operator == dag.get_task(ti.task_id).task_type
            if ti.task_id in task_ids and ti.execution_date in execution_dates:
                if map_task_pairs:
                    if (ti.task_id, ti.map_index) in map_task_pairs:
                        assert ti.state == state
                else:
                    assert ti.state == state, ti
                if ti.state in State.finished:
                    assert ti.end_date is not None, ti
            else:
                for old_ti in old_tis:
                    if old_ti.task_id == ti.task_id and old_ti.run_id == ti.run_id and (old_ti.map_index == ti.map_index):
                        assert ti.state == old_ti.state
                        break
                else:
                    unexpected_tis.append(ti)
        assert not unexpected_tis

    def test_mark_tasks_now(self):
        if False:
            while True:
                i = 10
        snapshot = TestMarkTasks.snapshot_state(self.dag1, self.execution_dates)
        task = self.dag1.get_task('runme_1')
        dr = DagRun.find(dag_id=self.dag1.dag_id, execution_date=self.execution_dates[0])[0]
        altered = set_state(tasks=[task], run_id=dr.run_id, upstream=False, downstream=False, future=False, past=False, state=State.SUCCESS, commit=False)
        assert len(altered) == 1
        self.verify_state(self.dag1, [task.task_id], [self.execution_dates[0]], None, snapshot)
        altered = set_state(tasks=[task], run_id=dr.run_id, upstream=False, downstream=False, future=False, past=False, state=State.SUCCESS, commit=True)
        assert len(altered) == 1
        self.verify_state(self.dag1, [task.task_id], [self.execution_dates[0]], State.SUCCESS, snapshot)
        altered = set_state(tasks=[task], run_id=dr.run_id, upstream=False, downstream=False, future=False, past=False, state=State.SUCCESS, commit=True)
        assert len(altered) == 0
        self.verify_state(self.dag1, [task.task_id], [self.execution_dates[0]], State.SUCCESS, snapshot)
        altered = set_state(tasks=[task], run_id=dr.run_id, upstream=False, downstream=False, future=False, past=False, state=State.FAILED, commit=True)
        assert len(altered) == 1
        self.verify_state(self.dag1, [task.task_id], [self.execution_dates[0]], State.FAILED, snapshot)
        snapshot = TestMarkTasks.snapshot_state(self.dag1, self.execution_dates)
        task = self.dag1.get_task('runme_0')
        altered = set_state(tasks=[task], run_id=dr.run_id, upstream=False, downstream=False, future=False, past=False, state=State.SUCCESS, commit=True)
        assert len(altered) == 1
        self.verify_state(self.dag1, [task.task_id], [self.execution_dates[0]], State.SUCCESS, snapshot)
        snapshot = TestMarkTasks.snapshot_state(self.dag3, self.dag3_execution_dates)
        task = self.dag3.get_task('run_this')
        dr = DagRun.find(dag_id=self.dag3.dag_id, execution_date=self.dag3_execution_dates[1])[0]
        altered = set_state(tasks=[task], run_id=dr.run_id, upstream=False, downstream=False, future=False, past=False, state=State.FAILED, commit=True)
        assert len(altered) == 1
        self.verify_state(self.dag3, [task.task_id], [self.dag3_execution_dates[1]], State.FAILED, snapshot)
        self.verify_state(self.dag3, [task.task_id], [self.dag3_execution_dates[0]], None, snapshot)
        self.verify_state(self.dag3, [task.task_id], [self.dag3_execution_dates[2]], None, snapshot)

    def test_mark_downstream(self):
        if False:
            while True:
                i = 10
        snapshot = TestMarkTasks.snapshot_state(self.dag1, self.execution_dates)
        task = self.dag1.get_task('runme_1')
        relatives = task.get_flat_relatives(upstream=False)
        task_ids = [t.task_id for t in relatives]
        task_ids.append(task.task_id)
        dr = DagRun.find(dag_id=self.dag1.dag_id, execution_date=self.execution_dates[0])[0]
        altered = set_state(tasks=[task], run_id=dr.run_id, upstream=False, downstream=True, future=False, past=False, state=State.SUCCESS, commit=True)
        assert len(altered) == 3
        self.verify_state(self.dag1, task_ids, [self.execution_dates[0]], State.SUCCESS, snapshot)

    def test_mark_upstream(self):
        if False:
            for i in range(10):
                print('nop')
        snapshot = TestMarkTasks.snapshot_state(self.dag1, self.execution_dates)
        task = self.dag1.get_task('run_after_loop')
        dr = DagRun.find(dag_id=self.dag1.dag_id, execution_date=self.execution_dates[0])[0]
        relatives = task.get_flat_relatives(upstream=True)
        task_ids = [t.task_id for t in relatives]
        task_ids.append(task.task_id)
        altered = set_state(tasks=[task], run_id=dr.run_id, upstream=True, downstream=False, future=False, past=False, state=State.SUCCESS, commit=True)
        assert len(altered) == 4
        self.verify_state(self.dag1, task_ids, [self.execution_dates[0]], State.SUCCESS, snapshot)

    def test_mark_tasks_future(self):
        if False:
            while True:
                i = 10
        snapshot = TestMarkTasks.snapshot_state(self.dag1, self.execution_dates)
        task = self.dag1.get_task('runme_1')
        dr = DagRun.find(dag_id=self.dag1.dag_id, execution_date=self.execution_dates[0])[0]
        altered = set_state(tasks=[task], run_id=dr.run_id, upstream=False, downstream=False, future=True, past=False, state=State.SUCCESS, commit=True)
        assert len(altered) == 2
        self.verify_state(self.dag1, [task.task_id], self.execution_dates, State.SUCCESS, snapshot)
        snapshot = TestMarkTasks.snapshot_state(self.dag3, self.dag3_execution_dates)
        task = self.dag3.get_task('run_this')
        dr = DagRun.find(dag_id=self.dag3.dag_id, execution_date=self.dag3_execution_dates[1])[0]
        altered = set_state(tasks=[task], run_id=dr.run_id, upstream=False, downstream=False, future=True, past=False, state=State.FAILED, commit=True)
        assert len(altered) == 2
        self.verify_state(self.dag3, [task.task_id], [self.dag3_execution_dates[0]], None, snapshot)
        self.verify_state(self.dag3, [task.task_id], self.dag3_execution_dates[1:], State.FAILED, snapshot)

    def test_mark_tasks_past(self):
        if False:
            while True:
                i = 10
        snapshot = TestMarkTasks.snapshot_state(self.dag1, self.execution_dates)
        task = self.dag1.get_task('runme_1')
        dr = DagRun.find(dag_id=self.dag1.dag_id, execution_date=self.execution_dates[1])[0]
        altered = set_state(tasks=[task], run_id=dr.run_id, upstream=False, downstream=False, future=False, past=True, state=State.SUCCESS, commit=True)
        assert len(altered) == 2
        self.verify_state(self.dag1, [task.task_id], self.execution_dates, State.SUCCESS, snapshot)
        snapshot = TestMarkTasks.snapshot_state(self.dag3, self.dag3_execution_dates)
        task = self.dag3.get_task('run_this')
        dr = DagRun.find(dag_id=self.dag3.dag_id, execution_date=self.dag3_execution_dates[1])[0]
        altered = set_state(tasks=[task], run_id=dr.run_id, upstream=False, downstream=False, future=False, past=True, state=State.FAILED, commit=True)
        assert len(altered) == 2
        self.verify_state(self.dag3, [task.task_id], self.dag3_execution_dates[:2], State.FAILED, snapshot)
        self.verify_state(self.dag3, [task.task_id], [self.dag3_execution_dates[2]], None, snapshot)

    def test_mark_tasks_multiple(self):
        if False:
            print('Hello World!')
        snapshot = TestMarkTasks.snapshot_state(self.dag1, self.execution_dates)
        tasks = [self.dag1.get_task('runme_1'), self.dag1.get_task('runme_2')]
        dr = DagRun.find(dag_id=self.dag1.dag_id, execution_date=self.execution_dates[0])[0]
        altered = set_state(tasks=tasks, run_id=dr.run_id, upstream=False, downstream=False, future=False, past=False, state=State.SUCCESS, commit=True)
        assert len(altered) == 2
        self.verify_state(self.dag1, [task.task_id for task in tasks], [self.execution_dates[0]], State.SUCCESS, snapshot)

    @pytest.mark.backend('sqlite', 'postgres')
    def test_mark_tasks_subdag(self):
        if False:
            return 10
        snapshot = TestMarkTasks.snapshot_state(self.dag2, self.execution_dates)
        task = self.dag2.get_task('section-1')
        relatives = task.get_flat_relatives(upstream=False)
        task_ids = [t.task_id for t in relatives]
        task_ids.append(task.task_id)
        dr = DagRun.find(dag_id=self.dag2.dag_id, execution_date=self.execution_dates[0])[0]
        altered = set_state(tasks=[task], run_id=dr.run_id, upstream=False, downstream=True, future=False, past=False, state=State.SUCCESS, commit=True)
        assert len(altered) == 14
        self.verify_state(self.dag2, task_ids, [self.execution_dates[0]], State.SUCCESS, snapshot)

    def test_mark_mapped_task_instance_state(self, session):
        if False:
            i = 10
            return i + 15
        mapped = self.dag4.get_task('consumer')
        tasks = [(mapped, 0), (mapped, 1)]
        dr = DagRun.find(dag_id=self.dag4.dag_id, execution_date=self.execution_dates[0], session=session)[0]
        expand_mapped_task(mapped, dr.run_id, 'make_arg_lists', length=3, session=session)
        snapshot = TestMarkTasks.snapshot_state(self.dag4, self.execution_dates)
        altered = set_state(tasks=tasks, run_id=dr.run_id, upstream=True, downstream=False, future=False, past=False, state=State.SUCCESS, commit=True, session=session)
        assert len(altered) == 3
        self.verify_state(self.dag4, ['consumer', 'make_arg_lists'], [self.execution_dates[0]], State.SUCCESS, snapshot, map_task_pairs=[(task.task_id, map_index) for (task, map_index) in tasks] + [('make_arg_lists', -1)], session=session)

class TestMarkDAGRun:
    INITIAL_TASK_STATES = {'runme_0': State.SUCCESS, 'runme_1': State.SKIPPED, 'runme_2': State.UP_FOR_RETRY, 'also_run_this': State.QUEUED, 'run_after_loop': State.RUNNING, 'run_this_last': State.FAILED}

    @classmethod
    def setup_class(cls):
        if False:
            while True:
                i = 10
        dagbag = models.DagBag(include_examples=True, read_dags_from_db=False)
        cls.dag1 = dagbag.dags['miscellaneous_test_dag']
        cls.dag1.sync_to_db()
        cls.dag2 = dagbag.dags['example_subdag_operator']
        cls.dag2.sync_to_db()
        cls.execution_dates = [timezone.datetime(2022, 1, 1), timezone.datetime(2022, 1, 2), timezone.datetime(2022, 1, 3)]

    def setup_method(self):
        if False:
            for i in range(10):
                print('nop')
        clear_db_runs()

    def teardown_method(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        clear_db_runs()

    def _get_num_tasks_with_starting_state(self, state: State, inclusion: bool):
        if False:
            print('Hello World!')
        '\n        If ``inclusion=True``, get num tasks with initial state ``state``.\n        Otherwise, get number tasks with initial state not equal to ``state``\n        :param state: State to compare against\n        :param inclusion: whether to look for inclusion or exclusion\n        :return: number of tasks meeting criteria\n        '
        states = self.INITIAL_TASK_STATES.values()

        def compare(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return x == y if inclusion else x != y
        return len([s for s in states if compare(s, state)])

    def _get_num_tasks_with_non_completed_state(self):
        if False:
            i = 10
            return i + 15
        '\n        Return the non completed tasks.\n        :return: number of tasks in non completed state (SUCCESS, FAILED, SKIPPED, UPSTREAM_FAILED)\n        '
        expected = len(self.INITIAL_TASK_STATES.values()) - self._get_num_tasks_with_starting_state(State.SUCCESS, inclusion=True)
        expected = expected - self._get_num_tasks_with_starting_state(State.FAILED, inclusion=True)
        expected = expected - self._get_num_tasks_with_starting_state(State.SKIPPED, inclusion=True)
        expected = expected - self._get_num_tasks_with_starting_state(State.UPSTREAM_FAILED, inclusion=True)
        return expected

    def _set_default_task_instance_states(self, dr):
        if False:
            i = 10
            return i + 15
        for (task_id, state) in self.INITIAL_TASK_STATES.items():
            dr.get_task_instance(task_id).set_state(state)

    def _verify_task_instance_states_remain_default(self, dr):
        if False:
            for i in range(10):
                print('nop')
        for (task_id, state) in self.INITIAL_TASK_STATES.items():
            assert dr.get_task_instance(task_id).state == state

    @provide_session
    def _verify_task_instance_states(self, dag, date, state, session=None):
        if False:
            print('Hello World!')
        TI = models.TaskInstance
        tis = session.query(TI).filter(TI.dag_id == dag.dag_id, TI.execution_date == date)
        for ti in tis:
            assert ti.state == state

    def _create_test_dag_run(self, state, date):
        if False:
            for i in range(10):
                print('nop')
        return self.dag1.create_dagrun(run_type=DagRunType.MANUAL, state=state, start_date=date, execution_date=date, data_interval=(date, date))

    def _verify_dag_run_state(self, dag, date, state):
        if False:
            print('Hello World!')
        drs = models.DagRun.find(dag_id=dag.dag_id, execution_date=date)
        dr = drs[0]
        assert dr.get_state() == state

    @provide_session
    def _verify_dag_run_dates(self, dag, date, state, middle_time, session=None):
        if False:
            return 10
        DR = DagRun
        dr = session.query(DR).filter(DR.dag_id == dag.dag_id, DR.execution_date == date).one()
        if state == State.RUNNING:
            assert dr.start_date > middle_time
            assert dr.end_date is None
        else:
            assert dr.start_date < middle_time
            assert dr.end_date > middle_time

    def test_set_running_dag_run_to_success(self):
        if False:
            while True:
                i = 10
        date = self.execution_dates[0]
        dr = self._create_test_dag_run(State.RUNNING, date)
        middle_time = timezone.utcnow()
        self._set_default_task_instance_states(dr)
        altered = set_dag_run_state_to_success(dag=self.dag1, run_id=dr.run_id, commit=True)
        expected = self._get_num_tasks_with_starting_state(State.SUCCESS, inclusion=False)
        assert len(altered) == expected
        self._verify_dag_run_state(self.dag1, date, State.SUCCESS)
        self._verify_task_instance_states(self.dag1, date, State.SUCCESS)
        self._verify_dag_run_dates(self.dag1, date, State.SUCCESS, middle_time)

    def test_set_running_dag_run_to_failed(self):
        if False:
            return 10
        date = self.execution_dates[0]
        dr = self._create_test_dag_run(State.RUNNING, date)
        middle_time = timezone.utcnow()
        self._set_default_task_instance_states(dr)
        altered = set_dag_run_state_to_failed(dag=self.dag1, run_id=dr.run_id, commit=True)
        expected = self._get_num_tasks_with_non_completed_state()
        assert len(altered) == expected
        self._verify_dag_run_state(self.dag1, date, State.FAILED)
        assert dr.get_task_instance('run_after_loop').state == State.FAILED
        self._verify_dag_run_dates(self.dag1, date, State.FAILED, middle_time)

    @pytest.mark.parametrize('dag_run_alter_function, new_state', [(set_dag_run_state_to_running, State.RUNNING), (set_dag_run_state_to_queued, State.QUEUED)])
    def test_set_running_dag_run_to_activate_state(self, dag_run_alter_function: Callable, new_state: State):
        if False:
            for i in range(10):
                print('nop')
        date = self.execution_dates[0]
        dr = self._create_test_dag_run(State.RUNNING, date)
        middle_time = timezone.utcnow()
        self._set_default_task_instance_states(dr)
        altered = dag_run_alter_function(dag=self.dag1, run_id=dr.run_id, commit=True)
        assert len(altered) == 0
        self._verify_dag_run_state(self.dag1, date, new_state)
        self._verify_task_instance_states_remain_default(dr)
        self._verify_dag_run_dates(self.dag1, date, new_state, middle_time)

    @pytest.mark.parametrize('completed_state', [State.SUCCESS, State.FAILED])
    def test_set_success_dag_run_to_success(self, completed_state):
        if False:
            return 10
        date = self.execution_dates[0]
        dr = self._create_test_dag_run(completed_state, date)
        middle_time = timezone.utcnow()
        self._set_default_task_instance_states(dr)
        altered = set_dag_run_state_to_success(dag=self.dag1, run_id=dr.run_id, commit=True)
        expected = self._get_num_tasks_with_starting_state(State.SUCCESS, inclusion=False)
        assert len(altered) == expected
        self._verify_dag_run_state(self.dag1, date, State.SUCCESS)
        self._verify_task_instance_states(self.dag1, date, State.SUCCESS)
        self._verify_dag_run_dates(self.dag1, date, State.SUCCESS, middle_time)

    @pytest.mark.parametrize('completed_state', [State.SUCCESS, State.FAILED])
    def test_set_completed_dag_run_to_failed(self, completed_state):
        if False:
            while True:
                i = 10
        date = self.execution_dates[0]
        dr = self._create_test_dag_run(completed_state, date)
        middle_time = timezone.utcnow()
        self._set_default_task_instance_states(dr)
        altered = set_dag_run_state_to_failed(dag=self.dag1, run_id=dr.run_id, commit=True)
        expected = self._get_num_tasks_with_non_completed_state()
        assert len(altered) == expected
        self._verify_dag_run_state(self.dag1, date, State.FAILED)
        assert dr.get_task_instance('run_after_loop').state == State.FAILED
        self._verify_dag_run_dates(self.dag1, date, State.FAILED, middle_time)

    @pytest.mark.parametrize('dag_run_alter_function,new_state', [(set_dag_run_state_to_running, State.RUNNING), (set_dag_run_state_to_queued, State.QUEUED)])
    def test_set_success_dag_run_to_activate_state(self, dag_run_alter_function: Callable, new_state: State):
        if False:
            while True:
                i = 10
        date = self.execution_dates[0]
        dr = self._create_test_dag_run(State.SUCCESS, date)
        middle_time = timezone.utcnow()
        self._set_default_task_instance_states(dr)
        altered = dag_run_alter_function(dag=self.dag1, run_id=dr.run_id, commit=True)
        assert len(altered) == 0
        self._verify_dag_run_state(self.dag1, date, new_state)
        self._verify_task_instance_states_remain_default(dr)
        self._verify_dag_run_dates(self.dag1, date, new_state, middle_time)

    @pytest.mark.parametrize('dag_run_alter_function,state', [(set_dag_run_state_to_running, State.RUNNING), (set_dag_run_state_to_queued, State.QUEUED)])
    def test_set_failed_dag_run_to_activate_state(self, dag_run_alter_function: Callable, state: State):
        if False:
            i = 10
            return i + 15
        date = self.execution_dates[0]
        dr = self._create_test_dag_run(State.SUCCESS, date)
        middle_time = timezone.utcnow()
        self._set_default_task_instance_states(dr)
        altered = dag_run_alter_function(dag=self.dag1, run_id=dr.run_id, commit=True)
        assert len(altered) == 0
        self._verify_dag_run_state(self.dag1, date, state)
        self._verify_task_instance_states_remain_default(dr)
        self._verify_dag_run_dates(self.dag1, date, state, middle_time)

    def test_set_state_without_commit(self):
        if False:
            while True:
                i = 10
        date = self.execution_dates[0]
        dr = self._create_test_dag_run(State.RUNNING, date)
        self._set_default_task_instance_states(dr)
        will_be_altered = set_dag_run_state_to_running(dag=self.dag1, run_id=dr.run_id, commit=False)
        assert len(will_be_altered) == 0
        self._verify_dag_run_state(self.dag1, date, State.RUNNING)
        self._verify_task_instance_states_remain_default(dr)
        will_be_altered = set_dag_run_state_to_queued(dag=self.dag1, run_id=dr.run_id, commit=False)
        assert len(will_be_altered) == 0
        self._verify_dag_run_state(self.dag1, date, State.RUNNING)
        self._verify_task_instance_states_remain_default(dr)
        will_be_altered = set_dag_run_state_to_failed(dag=self.dag1, run_id=dr.run_id, commit=False)
        expected = self._get_num_tasks_with_non_completed_state()
        assert len(will_be_altered) == expected
        self._verify_dag_run_state(self.dag1, date, State.RUNNING)
        self._verify_task_instance_states_remain_default(dr)
        will_be_altered = set_dag_run_state_to_success(dag=self.dag1, run_id=dr.run_id, commit=False)
        expected = self._get_num_tasks_with_starting_state(State.SUCCESS, inclusion=False)
        assert len(will_be_altered) == expected
        self._verify_dag_run_state(self.dag1, date, State.RUNNING)
        self._verify_task_instance_states_remain_default(dr)

    @provide_session
    def test_set_state_with_multiple_dagruns(self, session=None):
        if False:
            print('Hello World!')
        self.dag2.create_dagrun(run_type=DagRunType.MANUAL, state=State.FAILED, execution_date=self.execution_dates[0], data_interval=(self.execution_dates[0], self.execution_dates[0]), session=session)
        dr2 = self.dag2.create_dagrun(run_type=DagRunType.MANUAL, state=State.FAILED, execution_date=self.execution_dates[1], data_interval=(self.execution_dates[1], self.execution_dates[1]), session=session)
        self.dag2.create_dagrun(run_type=DagRunType.MANUAL, state=State.RUNNING, execution_date=self.execution_dates[2], data_interval=(self.execution_dates[2], self.execution_dates[2]), session=session)
        altered = set_dag_run_state_to_success(dag=self.dag2, run_id=dr2.run_id, commit=True)

        def count_dag_tasks(dag):
            if False:
                print('Hello World!')
            count = len(dag.tasks)
            subdag_counts = [count_dag_tasks(subdag) for subdag in dag.subdags]
            count += sum(subdag_counts)
            return count
        assert len(altered) == count_dag_tasks(self.dag2)
        self._verify_dag_run_state(self.dag2, self.execution_dates[1], State.SUCCESS)
        models.DagRun.find(dag_id=self.dag2.dag_id, execution_date=self.execution_dates[0])
        self._verify_dag_run_state(self.dag2, self.execution_dates[0], State.FAILED)
        models.DagRun.find(dag_id=self.dag2.dag_id, execution_date=self.execution_dates[2])
        self._verify_dag_run_state(self.dag2, self.execution_dates[2], State.RUNNING)

    def test_set_dag_run_state_edge_cases(self):
        if False:
            for i in range(10):
                print('nop')
        altered = set_dag_run_state_to_success(dag=None, execution_date=self.execution_dates[0])
        assert len(altered) == 0
        altered = set_dag_run_state_to_failed(dag=None, execution_date=self.execution_dates[0])
        assert len(altered) == 0
        altered = set_dag_run_state_to_running(dag=None, execution_date=self.execution_dates[0])
        assert len(altered) == 0
        altered = set_dag_run_state_to_queued(dag=None, execution_date=self.execution_dates[0])
        assert len(altered) == 0
        altered = set_dag_run_state_to_success(dag=self.dag1, run_id=None)
        assert len(altered) == 0
        altered = set_dag_run_state_to_failed(dag=self.dag1, run_id=None)
        assert len(altered) == 0
        altered = set_dag_run_state_to_running(dag=self.dag1, run_id=None)
        assert len(altered) == 0
        altered = set_dag_run_state_to_queued(dag=self.dag1, run_id=None)
        assert len(altered) == 0
        with pytest.raises(ValueError):
            set_dag_run_state_to_success(dag=self.dag2, run_id='dag_run_id_that_does_not_exist')
        with pytest.raises(ValueError):
            set_dag_run_state_to_success(dag=self.dag2, run_id='dag_run_id_that_does_not_exist')

    def test_set_dag_run_state_to_failed_no_running_tasks(self):
        if False:
            i = 10
            return i + 15
        '\n        set_dag_run_state_to_failed when there are no running tasks to update\n        '
        date = self.execution_dates[0]
        dr = self._create_test_dag_run(State.SUCCESS, date)
        for task in self.dag1.tasks:
            dr.get_task_instance(task.task_id).set_state(State.SUCCESS)
        set_dag_run_state_to_failed(dag=self.dag1, run_id=dr.run_id)