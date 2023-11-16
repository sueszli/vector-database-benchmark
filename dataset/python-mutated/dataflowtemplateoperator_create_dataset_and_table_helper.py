from google.cloud import bigquery
project = 'your-project'
location = 'US'
dataset_name = 'average_weather'

def create_dataset_and_table(project, location, dataset_name):
    if False:
        print('Hello World!')
    client = bigquery.Client(project)
    dataset_id = f'{project}.{dataset_name}'
    dataset = bigquery.Dataset(dataset_id)
    dataset.location = location
    dataset = client.create_dataset(dataset)
    print(f'Created dataset {client.project}.{dataset.dataset_id}')
    table_id = f'{client.project}.{dataset_name}.average_weather'
    schema = [bigquery.SchemaField('location', 'GEOGRAPHY', mode='REQUIRED'), bigquery.SchemaField('average_temperature', 'INTEGER', mode='REQUIRED'), bigquery.SchemaField('month', 'STRING', mode='REQUIRED'), bigquery.SchemaField('inches_of_rain', 'NUMERIC', mode='NULLABLE'), bigquery.SchemaField('is_current', 'BOOLEAN', mode='NULLABLE'), bigquery.SchemaField('latest_measurement', 'DATE', mode='NULLABLE')]
    table = bigquery.Table(table_id, schema=schema)
    table = client.create_table(table)
    print(f'Created table {table.project}.{table.dataset_id}.{table.table_id}')
    return (dataset, table)
if __name__ == '__main__':
    create_dataset_and_table(project, location, 'average_weather')