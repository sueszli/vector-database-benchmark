import quickstart_getfeed

def test_get_feed(capsys, test_feed):
    if False:
        return 10
    quickstart_getfeed.get_feed(test_feed.name)
    (out, _) = capsys.readouterr()
    assert 'gotten_feed' in out