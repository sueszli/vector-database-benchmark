from __future__ import annotations
import os
from datetime import datetime
from airflow.decorators import task
from airflow.models.dag import DAG
from airflow.providers.influxdb.hooks.influxdb import InfluxDBHook

@task(task_id='influxdb_task')
def test_influxdb_hook():
    if False:
        i = 10
        return i + 15
    bucket_name = 'test-influx'
    influxdb_hook = InfluxDBHook()
    client = influxdb_hook.get_conn()
    print(client)
    print(f'Organization name {influxdb_hook.org_name}')
    influxdb_hook.create_bucket(bucket_name, 'Bucket to test influxdb connection', influxdb_hook.org_name)
    influxdb_hook.write(bucket_name, 'test_point', 'location', 'Prague', 'temperature', 25.3, True)
    tables = influxdb_hook.query('from(bucket:"test-influx") |> range(start: -10m)')
    for table in tables:
        print(table)
        for record in table.records:
            print(record.values)
    bucket_id = influxdb_hook.find_bucket_id_by_name(bucket_name)
    print(bucket_id)
    influxdb_hook.delete_bucket(bucket_name)
ENV_ID = os.environ.get('SYSTEM_TESTS_ENV_ID')
DAG_ID = 'influxdb_example_dag'
with DAG(dag_id=DAG_ID, schedule=None, start_date=datetime(2021, 1, 1), max_active_runs=1, tags=['example']) as dag:
    test_influxdb_hook()
    from tests.system.utils.watcher import watcher
    list(dag.tasks) >> watcher()
from tests.system.utils import get_test_run
test_run = get_test_run(dag)