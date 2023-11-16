import lookup_entry_sql_resource
BIGQUERY_PROJECT = 'bigquery-public-data'
BIGQUERY_DATASET = 'new_york_taxi_trips'

def test_lookup_entry():
    if False:
        for i in range(10):
            print('nop')
    sql_name = f'bigquery.dataset.`{BIGQUERY_PROJECT}`.`{BIGQUERY_DATASET}`'
    resource_name = f'//bigquery.googleapis.com/projects/{BIGQUERY_PROJECT}/datasets/{BIGQUERY_DATASET}'
    entry = lookup_entry_sql_resource.sample_lookup_entry(sql_name)
    assert entry.linked_resource == resource_name