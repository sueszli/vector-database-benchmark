import os
import uuid
import quickstart_create_saved_query
PROJECT = os.environ['GOOGLE_CLOUD_PROJECT']
SAVED_QUERY_ID = f'saved-query-{uuid.uuid4().hex}'

def test_create_saved_query(capsys, saved_query_deleter):
    if False:
        while True:
            i = 10
    saved_query = quickstart_create_saved_query.create_saved_query(PROJECT, SAVED_QUERY_ID, 'saved query foo')
    saved_query_deleter.append(saved_query.name)
    expected_resource_name_suffix = f'savedQueries/{SAVED_QUERY_ID}'
    assert saved_query.name.endswith(expected_resource_name_suffix)