import os
from aggregate_query_count import create_count_query
os.environ['GOOGLE_CLOUD_PROJECT'] = os.environ['FIRESTORE_PROJECT']
PROJECT_ID = os.environ['GOOGLE_CLOUD_PROJECT']

def test_create_count_query(capsys):
    if False:
        i = 10
        return i + 15
    create_count_query(project_id=PROJECT_ID)
    (out, _) = capsys.readouterr()
    assert 'Alias of results from query: all' in out
    assert 'Number of' in out