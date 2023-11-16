from __future__ import annotations
import pendulum
import pytest
from airflow.models import DagBag
from airflow.operators.empty import EmptyOperator
from airflow.utils.state import DagRunState, TaskInstanceState
from airflow.utils.types import DagRunType
from tests.test_utils.db import clear_db_runs
pytestmark = pytest.mark.db_test

@pytest.fixture(autouse=True, scope='module')
def examples_dag_bag():
    if False:
        i = 10
        return i + 15
    return DagBag(include_examples=False, read_dags_from_db=True)

@pytest.fixture(autouse=True)
def clean():
    if False:
        for i in range(10):
            print('nop')
    clear_db_runs()
    yield
    clear_db_runs()

@pytest.fixture
def freeze_time_for_dagruns(time_machine):
    if False:
        for i in range(10):
            print('nop')
    time_machine.move_to('2023-05-02T00:00:00+00:00', tick=False)
    yield

@pytest.fixture
def make_dag_runs(dag_maker, session, time_machine):
    if False:
        while True:
            i = 10
    with dag_maker(dag_id='test_dag_id', serialized=True, session=session, start_date=pendulum.DateTime(2023, 2, 1, 0, 0, 0, tzinfo=pendulum.UTC)):
        EmptyOperator(task_id='task_1') >> EmptyOperator(task_id='task_2')
    date = dag_maker.dag.start_date
    run1 = dag_maker.create_dagrun(run_id='run_1', state=DagRunState.SUCCESS, run_type=DagRunType.SCHEDULED, execution_date=date, start_date=date)
    run2 = dag_maker.create_dagrun(run_id='run_2', state=DagRunState.FAILED, run_type=DagRunType.DATASET_TRIGGERED, execution_date=dag_maker.dag.next_dagrun_info(date).logical_date, start_date=dag_maker.dag.next_dagrun_info(date).logical_date)
    run3 = dag_maker.create_dagrun(run_id='run_3', state=DagRunState.RUNNING, run_type=DagRunType.SCHEDULED, execution_date=pendulum.DateTime(2023, 2, 3, 0, 0, 0, tzinfo=pendulum.UTC), start_date=pendulum.DateTime(2023, 2, 3, 0, 0, 0, tzinfo=pendulum.UTC))
    run3.end_date = None
    for ti in run1.task_instances:
        ti.state = TaskInstanceState.SUCCESS
    for ti in run2.task_instances:
        ti.state = TaskInstanceState.FAILED
    time_machine.move_to('2023-07-02T00:00:00+00:00', tick=False)
    session.flush()

@pytest.mark.usefixtures('freeze_time_for_dagruns', 'make_dag_runs')
def test_historical_metrics_data(admin_client, session, time_machine):
    if False:
        for i in range(10):
            print('nop')
    resp = admin_client.get('/object/historical_metrics_data?start_date=2023-01-01T00:00&end_date=2023-08-02T00:00', follow_redirects=True)
    assert resp.status_code == 200
    assert resp.json == {'dag_run_states': {'failed': 1, 'queued': 0, 'running': 1, 'success': 1}, 'dag_run_types': {'backfill': 0, 'dataset_triggered': 1, 'manual': 0, 'scheduled': 2}, 'task_instance_states': {'deferred': 0, 'failed': 2, 'no_status': 2, 'queued': 0, 'removed': 0, 'restarting': 0, 'running': 0, 'scheduled': 0, 'shutdown': 0, 'skipped': 0, 'success': 2, 'up_for_reschedule': 0, 'up_for_retry': 0, 'upstream_failed': 0}}

@pytest.mark.usefixtures('freeze_time_for_dagruns', 'make_dag_runs')
def test_historical_metrics_data_date_filters(admin_client, session):
    if False:
        while True:
            i = 10
    resp = admin_client.get('/object/historical_metrics_data?start_date=2023-02-02T00:00&end_date=2023-06-02T00:00', follow_redirects=True)
    assert resp.status_code == 200
    assert resp.json == {'dag_run_states': {'failed': 1, 'queued': 0, 'running': 0, 'success': 0}, 'dag_run_types': {'backfill': 0, 'dataset_triggered': 1, 'manual': 0, 'scheduled': 0}, 'task_instance_states': {'deferred': 0, 'failed': 2, 'no_status': 0, 'queued': 0, 'removed': 0, 'restarting': 0, 'running': 0, 'scheduled': 0, 'shutdown': 0, 'skipped': 0, 'success': 0, 'up_for_reschedule': 0, 'up_for_retry': 0, 'upstream_failed': 0}}