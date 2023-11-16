from airflow import models
from airflow.hooks.base import BaseHook
from airflow.providers.google.cloud.operators.bigquery import BigQueryCheckOperator
from airflow.providers.google.cloud.operators.dataflow import DataflowTemplatedJobStartOperator
from airflow.providers.google.cloud.sensors.gcs import GCSObjectExistenceSensor
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
from airflow.utils.dates import days_ago
from airflow.utils.state import State
BUCKET_NAME = 'cloud-samples-data/composer/data-orchestration-blog-example'
DATA_FILE_NAME = 'bike_station_data.csv'
PROJECT_ID = '{{var.value.gcp_project}}'
DATASET = '{{var.value.bigquery_dataset}}'
TABLE = '{{var.value.bigquery_table}}'

def on_failure_callback(context):
    if False:
        i = 10
        return i + 15
    ti = context.get('task_instance')
    slack_msg = f"\n            :red_circle: Task Failed.\n            *Task*: {ti.task_id}\n            *Dag*: {ti.dag_id}\n            *Execution Time*: {context.get('execution_date')}\n            *Log Url*: {ti.log_url}\n            "
    slack_webhook_token = BaseHook.get_connection('slack_connection').password
    slack_error = SlackWebhookOperator(task_id='post_slack_error', http_conn_id='slack_connection', channel='#airflow-alerts', webhook_token=slack_webhook_token, message=slack_msg)
    slack_error.execute(context)
with models.DAG('dataflow_to_bq_workflow', schedule_interval=None, start_date=days_ago(1), default_args={'on_failure_callback': on_failure_callback}) as dag:
    validate_file_exists = GCSObjectExistenceSensor(task_id='validate_file_exists', bucket=BUCKET_NAME, object=DATA_FILE_NAME)
    start_dataflow_job = DataflowTemplatedJobStartOperator(task_id='start-dataflow-template-job', job_name='csv_to_bq_transform', template='gs://dataflow-templates/latest/GCS_Text_to_BigQuery', parameters={'javascriptTextTransformFunctionName': 'transform', 'javascriptTextTransformGcsPath': f'gs://{BUCKET_NAME}/udf_transform.js', 'JSONPath': f'gs://{BUCKET_NAME}/bq_schema.json', 'inputFilePattern': f'gs://{BUCKET_NAME}/{DATA_FILE_NAME}', 'bigQueryLoadingTemporaryDirectory': f'gs://{BUCKET_NAME}/tmp/', 'outputTable': f'{PROJECT_ID}:{DATASET}.{TABLE}'})
    execute_bigquery_sql = BigQueryCheckOperator(task_id='execute_bigquery_sql', sql=f'SELECT COUNT(*) FROM `{PROJECT_ID}.{DATASET}.{TABLE}`', use_legacy_sql=False)
    validate_file_exists >> start_dataflow_job >> execute_bigquery_sql
if __name__ == '__main__':
    dag.clear(dag_run_state=State.NONE)
    dag.run()