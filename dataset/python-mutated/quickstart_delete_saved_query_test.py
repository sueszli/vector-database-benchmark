import os
import quickstart_delete_saved_query
PROJECT = os.environ['GOOGLE_CLOUD_PROJECT']

def test_delete_saved_query(capsys, test_saved_query):
    if False:
        for i in range(10):
            print('nop')
    quickstart_delete_saved_query.delete_saved_query(test_saved_query.name)
    (out, _) = capsys.readouterr()
    assert 'deleted_saved_query' in out