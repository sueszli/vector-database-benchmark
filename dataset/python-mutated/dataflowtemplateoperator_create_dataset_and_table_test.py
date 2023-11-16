import os
import uuid
from google.cloud import bigquery
from . import dataflowtemplateoperator_create_dataset_and_table_helper as helper
PROJECT_ID = os.environ['GOOGLE_CLOUD_PROJECT']
client = bigquery.Client()
dataset_UUID = str(uuid.uuid4()).split('-')[0]
expected_schema = [bigquery.SchemaField('location', 'GEOGRAPHY', mode='REQUIRED'), bigquery.SchemaField('average_temperature', 'INTEGER', mode='REQUIRED'), bigquery.SchemaField('month', 'STRING', mode='REQUIRED'), bigquery.SchemaField('inches_of_rain', 'NUMERIC', mode='NULLABLE'), bigquery.SchemaField('is_current', 'BOOLEAN', mode='NULLABLE'), bigquery.SchemaField('latest_measurement', 'DATE', mode='NULLABLE')]

def test_creation():
    if False:
        for i in range(10):
            print('nop')
    try:
        (dataset, table) = helper.create_dataset_and_table(PROJECT_ID, 'US', dataset_UUID)
        assert table.table_id == 'average_weather'
        assert dataset.dataset_id == dataset_UUID
        assert table.schema == expected_schema
    finally:
        client.delete_dataset(dataset, delete_contents=True, not_found_ok=True)
        client.delete_table(table, not_found_ok=True)