import pytest
from pony.orm import db_session
from tribler.core.utilities.tracker_utils import MalformedTrackerURLException

@db_session
def test_create_tracker_state(metadata_store):
    if False:
        print('Hello World!')
    ts = metadata_store.TrackerState(url='http://tracker.tribler.org:80/announce')
    assert list(metadata_store.TrackerState.select())[0] == ts

@db_session
def test_canonicalize_tracker_state(metadata_store):
    if False:
        for i in range(10):
            print('nop')
    ts = metadata_store.TrackerState(url='http://tracker.tribler.org:80/announce/')
    assert metadata_store.TrackerState.get(url='http://tracker.tribler.org/announce') == ts

@db_session
def test_canonicalize_raise_on_malformed_url(metadata_store):
    if False:
        while True:
            i = 10
    with pytest.raises(MalformedTrackerURLException):
        metadata_store.TrackerState(url='udp://tracker.tribler.org/announce/')