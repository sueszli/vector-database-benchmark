"""Example Airflow DAG that performs an export from BQ tables listed in
config file to GCS, copies GCS objects across locations (e.g., from US to
EU) then imports from GCS to BQ. The DAG imports the gcs_to_gcs operator
from plugins and dynamically builds the tasks based on the list of tables.
Lastly, the DAG defines a specific application logger to generate logs.

This DAG relies on three Airflow variables
https://airflow.apache.org/docs/apache-airflow/stable/concepts/variables.html:
* table_list_file_path - CSV file listing source and target tables, including
Datasets.
* gcs_source_bucket - Google Cloud Storage bucket to use for exporting
BigQuery tables in source.
* gcs_dest_bucket - Google Cloud Storage bucket to use for importing
BigQuery tables in destination.
See https://cloud.google.com/storage/docs/creating-buckets for creating a
bucket.
"""
import csv
import datetime
import logging
from airflow import models
from airflow.operators import dummy
from airflow.providers.google.cloud.transfers import bigquery_to_gcs
from airflow.providers.google.cloud.transfers import gcs_to_bigquery
from airflow.providers.google.cloud.transfers import gcs_to_gcs
yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
default_args = {'owner': 'airflow', 'start_date': yesterday, 'depends_on_past': False, 'email': [''], 'email_on_failure': False, 'email_on_retry': False, 'retries': 1, 'retry_delay': datetime.timedelta(minutes=5)}
source_bucket = '{{var.value.gcs_source_bucket}}'
dest_bucket = '{{var.value.gcs_dest_bucket}}'
logger = logging.getLogger('bq_copy_us_to_eu_01')

def read_table_list(table_list_file):
    if False:
        for i in range(10):
            print('nop')
    "\n    Reads the table list file that will help in creating Airflow tasks in\n    the DAG dynamically.\n    :param table_list_file: (String) The file location of the table list file,\n    e.g. '/home/airflow/framework/table_list.csv'\n    :return table_list: (List) List of tuples containing the source and\n    target tables.\n    "
    table_list = []
    logger.info('Reading table_list_file from : %s' % str(table_list_file))
    try:
        with open(table_list_file, encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)
            for row in csv_reader:
                logger.info(row)
                table_tuple = {'table_source': row[0], 'table_dest': row[1]}
                table_list.append(table_tuple)
            return table_list
    except OSError as e:
        logger.error('Error opening table_list_file %s: ' % str(table_list_file), e)
with models.DAG('composer_sample_bq_copy_across_locations', default_args=default_args, schedule_interval=None) as dag:
    start = dummy.DummyOperator(task_id='start', trigger_rule='all_success')
    end = dummy.DummyOperator(task_id='end', trigger_rule='all_success')
    table_list_file_path = models.Variable.get('table_list_file_path')
    all_records = read_table_list(table_list_file_path)
    for record in all_records:
        logger.info(f'Generating tasks to transfer table: {record}')
        table_source = record['table_source']
        table_dest = record['table_dest']
        BQ_to_GCS = bigquery_to_gcs.BigQueryToGCSOperator(task_id='{}_BQ_to_GCS'.format(table_source.replace(':', '_')), source_project_dataset_table=table_source, destination_cloud_storage_uris=['{}-*.avro'.format('gs://' + source_bucket + '/' + table_source)], export_format='AVRO')
        GCS_to_GCS = gcs_to_gcs.GCSToGCSOperator(task_id='{}_GCS_to_GCS'.format(table_source.replace(':', '_')), source_bucket=source_bucket, source_object=f'{table_source}-*.avro', destination_bucket=dest_bucket)
        GCS_to_BQ = gcs_to_bigquery.GCSToBigQueryOperator(task_id='{}_GCS_to_BQ'.format(table_dest.replace(':', '_')), bucket=dest_bucket, source_objects=[f'{table_source}-*.avro'], destination_project_dataset_table=table_dest, source_format='AVRO', write_disposition='WRITE_TRUNCATE', autodetect=True)
        start >> BQ_to_GCS >> GCS_to_GCS >> GCS_to_BQ >> end