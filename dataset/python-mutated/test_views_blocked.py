from __future__ import annotations
import pytest
from airflow.models import DagModel
from airflow.models.dagbag import DagBag
from airflow.models.serialized_dag import SerializedDagModel
from airflow.operators.empty import EmptyOperator
from airflow.operators.subdag import SubDagOperator
from airflow.utils import timezone
from airflow.utils.session import create_session
from airflow.utils.state import State
from tests.test_utils.db import clear_db_runs
pytestmark = pytest.mark.db_test

@pytest.fixture()
def running_subdag(admin_client, dag_maker):
    if False:
        for i in range(10):
            print('nop')
    with dag_maker(dag_id='running_dag.subdag') as subdag:
        EmptyOperator(task_id='empty')
    with pytest.deprecated_call(), dag_maker(dag_id='running_dag') as dag:
        SubDagOperator(task_id='subdag', subdag=subdag)
    dag_bag = DagBag(include_examples=False)
    dag_bag.bag_dag(dag, root_dag=dag)
    with create_session() as session:
        dag_bag.sync_to_db(session=session)
        logical_date = timezone.datetime(2016, 1, 1)
        subdag.create_dagrun(run_id='blocked_run_example_bash_operator', state=State.RUNNING, execution_date=logical_date, data_interval=(logical_date, logical_date), start_date=timezone.datetime(2016, 1, 1), session=session)
        session.query(DagModel).filter(DagModel.dag_id == dag.dag_id).delete()
        session.query(SerializedDagModel).filter(SerializedDagModel.dag_id == dag.dag_id).delete()
    yield subdag
    with create_session() as session:
        session.query(DagModel).filter(DagModel.dag_id == subdag.dag_id).delete()
    clear_db_runs()

def test_blocked_subdag_success(admin_client, running_subdag):
    if False:
        i = 10
        return i + 15
    'Test the /blocked endpoint works when a DAG is deleted.\n\n    When a DAG is bagged, it is written to both DagModel and SerializedDagModel,\n    but its subdags are only written to DagModel (without serialization). Thus,\n    ``DagBag.get_dag(subdag_id)`` would raise ``SerializedDagNotFound`` if the\n    subdag was not previously bagged in the dagbag (perhaps due to its root DAG\n    being deleted). ``DagBag.get_dag()`` calls should catch the exception and\n    properly handle this situation.\n    '
    resp = admin_client.post('/blocked', data={'dag_ids': [running_subdag.dag_id]})
    assert resp.status_code == 200
    assert resp.json == [{'dag_id': running_subdag.dag_id, 'active_dag_run': 1, 'max_active_runs': 0}]