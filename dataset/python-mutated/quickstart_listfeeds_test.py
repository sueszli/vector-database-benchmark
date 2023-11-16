import os
import quickstart_listfeeds
PROJECT = os.environ['GOOGLE_CLOUD_PROJECT']

def test_list_feeds(capsys):
    if False:
        while True:
            i = 10
    parent_resource = f'projects/{PROJECT}'
    quickstart_listfeeds.list_feeds(parent_resource)
    (out, _) = capsys.readouterr()
    assert 'feeds' in out