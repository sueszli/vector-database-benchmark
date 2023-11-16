from __future__ import annotations
import json
import os
from datetime import datetime
from airflow.decorators import task
from airflow.models import Connection
from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator
from airflow.providers.google.cloud.operators.gcs import GCSCreateBucketOperator, GCSDeleteBucketOperator
from airflow.providers.google.cloud.transfers.sheets_to_gcs import GoogleSheetsToGCSOperator
from airflow.providers.google.suite.operators.sheets import GoogleSheetsCreateSpreadsheetOperator
from airflow.settings import Session
from airflow.utils.trigger_rule import TriggerRule
ENV_ID = os.environ.get('SYSTEM_TESTS_ENV_ID', 'default')
PROJECT_ID = os.environ.get('SYSTEM_TESTS_GCP_PROJECT', 'default')
DAG_ID = 'example_sheets_to_gcs'
BUCKET_NAME = f'bucket_{DAG_ID}_{ENV_ID}'
SPREADSHEET = {'properties': {'title': 'Test1'}, 'sheets': [{'properties': {'title': 'Sheet1'}}]}
CONNECTION_ID = f'connection_{DAG_ID}_{ENV_ID}'
with DAG(DAG_ID, schedule='@once', start_date=datetime(2021, 1, 1), catchup=False, tags=['example', 'sheets']) as dag:
    create_bucket = GCSCreateBucketOperator(task_id='create_bucket', bucket_name=BUCKET_NAME, project_id=PROJECT_ID)

    @task
    def create_temp_sheets_connection():
        if False:
            return 10
        conn = Connection(conn_id=CONNECTION_ID, conn_type='google_cloud_platform')
        conn_extra = {'scope': 'https://www.googleapis.com/auth/spreadsheets,https://www.googleapis.com/auth/cloud-platform', 'project': PROJECT_ID, 'keyfile_dict': ''}
        conn_extra_json = json.dumps(conn_extra)
        conn.set_extra(conn_extra_json)
        session = Session()
        session.add(conn)
        session.commit()
    create_temp_sheets_connection_task = create_temp_sheets_connection()
    create_spreadsheet = GoogleSheetsCreateSpreadsheetOperator(task_id='create_spreadsheet', spreadsheet=SPREADSHEET, gcp_conn_id=CONNECTION_ID)
    upload_sheet_to_gcs = GoogleSheetsToGCSOperator(task_id='upload_sheet_to_gcs', destination_bucket=BUCKET_NAME, spreadsheet_id="{{ task_instance.xcom_pull(task_ids='create_spreadsheet', key='spreadsheet_id') }}", gcp_conn_id=CONNECTION_ID)
    delete_temp_sheets_connection_task = BashOperator(task_id='delete_temp_sheets_connection', bash_command=f'airflow connections delete {CONNECTION_ID}', trigger_rule=TriggerRule.ALL_DONE)
    delete_bucket = GCSDeleteBucketOperator(task_id='delete_bucket', bucket_name=BUCKET_NAME, trigger_rule=TriggerRule.ALL_DONE)
    [create_bucket, create_temp_sheets_connection_task] >> create_spreadsheet >> upload_sheet_to_gcs >> [delete_bucket, delete_temp_sheets_connection_task]
    from tests.system.utils.watcher import watcher
    list(dag.tasks) >> watcher()
from tests.system.utils import get_test_run
test_run = get_test_run(dag)