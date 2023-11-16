import os
import tempfile
from airflow.models import DagBag, Variable
from dagster_airflow import make_dagster_job_from_airflow_dag, make_ephemeral_airflow_db_resource
from dagster_airflow_tests.marks import requires_local_db
DAG_RUN_CONF_DAG = '\nfrom airflow import models\n\nfrom airflow.operators.python_operator import PythonOperator\nfrom airflow.models import Variable\nimport datetime\n\ndefault_args = {"start_date": datetime.datetime(2023, 2, 1)}\n\nwith models.DAG(\n    dag_id="dag_run_conf_dag", default_args=default_args, schedule_interval=\'0 0 * * *\',\n) as dag_run_conf_dag:\n    def test_function(**kwargs):\n        Variable.set("CONFIGURATION_VALUE", kwargs[\'config_value\'])\n\n    PythonOperator(\n        task_id="previous_macro_test",\n        python_callable=test_function,\n        provide_context=True,\n        op_kwargs={\'config_value\': \'{{dag_run.conf.get("configuration_key")}}\'}\n    )\n'

@requires_local_db
def test_dag_run_conf_local() -> None:
    if False:
        return 10
    with tempfile.TemporaryDirectory() as dags_path:
        with open(os.path.join(dags_path, 'dag.py'), 'wb') as f:
            f.write(bytes(DAG_RUN_CONF_DAG.encode('utf-8')))
        airflow_db = make_ephemeral_airflow_db_resource(dag_run_config={'configuration_key': 'foo'})
        dag_bag = DagBag(dag_folder=dags_path)
        retry_dag = dag_bag.get_dag(dag_id='dag_run_conf_dag')
        job = make_dagster_job_from_airflow_dag(dag=retry_dag, resource_defs={'airflow_db': airflow_db})
        result = job.execute_in_process()
        assert result.success
        assert Variable.get('CONFIGURATION_VALUE') == 'foo'