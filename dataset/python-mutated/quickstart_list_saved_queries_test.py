import os
import quickstart_list_saved_queries
PROJECT = os.environ['GOOGLE_CLOUD_PROJECT']

def test_list_saved_queries(capsys):
    if False:
        i = 10
        return i + 15
    parent_resource = f'projects/{PROJECT}'
    quickstart_list_saved_queries.list_saved_queries(parent_resource)
    (out, _) = capsys.readouterr()
    assert 'saved_queries' in out