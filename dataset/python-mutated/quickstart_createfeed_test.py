import os
import uuid
import quickstart_createfeed
PROJECT = os.environ['GOOGLE_CLOUD_PROJECT']
ASSET_NAME = f'assets-{uuid.uuid4().hex}'
FEED_ID = f'feed-{uuid.uuid4().hex}'
FEED_ID_R = f'feed-{uuid.uuid4().hex}'

def test_create_feed(capsys, test_topic, deleter):
    if False:
        while True:
            i = 10
    from google.cloud import asset_v1
    feed = quickstart_createfeed.create_feed(PROJECT, FEED_ID, [ASSET_NAME], test_topic.name, asset_v1.ContentType.RESOURCE)
    deleter.append(feed.name)
    (out, _) = capsys.readouterr()
    assert 'feed' in out
    feed_r = quickstart_createfeed.create_feed(PROJECT, FEED_ID_R, [ASSET_NAME], test_topic.name, asset_v1.ContentType.RELATIONSHIP)
    deleter.append(feed_r.name)
    (out_r, _) = capsys.readouterr()
    assert 'feed' in out_r