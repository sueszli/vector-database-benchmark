import re
import lookup_entry
BIGQUERY_PROJECT = 'bigquery-public-data'
BIGQUERY_DATASET = 'new_york_taxi_trips'
BIGQUERY_TABLE = 'taxi_zone_geom'
PUBSUB_PROJECT = 'pubsub-public-data'
PUBSUB_TOPIC = 'taxirides-realtime'

def test_lookup_entry(capsys):
    if False:
        return 10
    override_values = {'bigquery_project_id': BIGQUERY_PROJECT, 'dataset_id': BIGQUERY_DATASET, 'table_id': BIGQUERY_TABLE, 'pubsub_project_id': PUBSUB_PROJECT, 'topic_id': PUBSUB_TOPIC}
    dataset_resource = f'//bigquery.googleapis.com/projects/{BIGQUERY_PROJECT}/datasets/{BIGQUERY_DATASET}'
    table_resource = f'//bigquery.googleapis.com/projects/{BIGQUERY_PROJECT}/datasets/{BIGQUERY_DATASET}/tables/{BIGQUERY_TABLE}'
    topic_resource = f'//pubsub.googleapis.com/projects/{PUBSUB_PROJECT}/topics/{PUBSUB_TOPIC}'
    lookup_entry.lookup_entry(override_values)
    (out, err) = capsys.readouterr()
    assert re.search(f'(Retrieved entry .+ for BigQuery Dataset resource {dataset_resource})', out)
    assert re.search(f'(Retrieved entry .+ for BigQuery Table resource {table_resource})', out)
    assert re.search(f'(Retrieved entry .+ for Pub/Sub Topic resource {topic_resource})', out)