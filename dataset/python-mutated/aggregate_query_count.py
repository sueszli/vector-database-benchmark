"""
Creates an aggregate query (COUNT) that returns the number of results in the query.

See https://cloud.google.com/python/docs/reference/firestore/latest before running this sample.
"""
from google.cloud import firestore
from google.cloud.firestore_v1 import aggregation
from google.cloud.firestore_v1.base_query import FieldFilter

def create_count_query(project_id: str) -> None:
    if False:
        i = 10
        return i + 15
    'Builds an aggregate query that returns the number of results in the query.\n\n    Arguments:\n      project_id: your Google Cloud Project ID\n    '
    client = firestore.Client(project=project_id)
    collection_ref = client.collection('users')
    query = collection_ref.where(filter=FieldFilter('born', '>', 1800))
    aggregate_query = aggregation.AggregationQuery(query)
    aggregate_query.count(alias='all')
    results = aggregate_query.get()
    for result in results:
        print(f'Alias of results from query: {result[0].alias}')
        print(f'Number of results from query: {result[0].value}')