import lookup_entry
BIGQUERY_PROJECT = 'bigquery-public-data'
BIGQUERY_DATASET = 'new_york_taxi_trips'

def test_lookup_entry(client, entry, project_id):
    if False:
        while True:
            i = 10
    bigquery_dataset = f'projects/{BIGQUERY_PROJECT}/datasets/{BIGQUERY_DATASET}'
    resource_name = f'//bigquery.googleapis.com/{bigquery_dataset}'
    found_entry = lookup_entry.sample_lookup_entry(resource_name)
    assert found_entry.linked_resource == resource_name