import quickstart_updatefeed

def test_update_feed(capsys, test_feed, another_topic):
    if False:
        while True:
            i = 10
    quickstart_updatefeed.update_feed(test_feed.name, another_topic.name)
    (out, _) = capsys.readouterr()
    assert 'updated_feed' in out