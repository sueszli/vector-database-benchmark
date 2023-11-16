"""
Builds a union (OR) query filter.

See https://cloud.google.com/python/docs/reference/datastore/latest before running code.
"""
from google.cloud import datastore
from google.cloud.datastore import query

def query_filter_or(project_id: str) -> None:
    if False:
        return 10
    'Builds a union of two queries (OR) filter.\n\n    Arguments:\n        project_id: your Google Cloud Project ID\n    '
    client = datastore.Client(project=project_id)
    or_query = client.query(kind='Task')
    or_filter = query.Or([query.PropertyFilter('description', '=', 'Buy milk'), query.PropertyFilter('description', '=', 'Feed cats')])
    or_query.add_filter(filter=or_filter)
    results = or_query.fetch()
    for result in results:
        print(result['description'])