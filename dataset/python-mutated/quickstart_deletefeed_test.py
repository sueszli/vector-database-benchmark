import os
import quickstart_deletefeed
PROJECT = os.environ['GOOGLE_CLOUD_PROJECT']

def test_delete_feed(capsys, test_feed):
    if False:
        while True:
            i = 10
    quickstart_deletefeed.delete_feed(test_feed.name)
    (out, _) = capsys.readouterr()
    assert 'deleted_feed' in out