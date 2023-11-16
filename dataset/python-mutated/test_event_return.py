import pytest
from pytestshellutils.utils.processes import terminate_process
import salt.utils.event
import salt.utils.stringutils

@pytest.mark.slow_test
def test_event_return(master_opts):
    if False:
        while True:
            i = 10
    evt = None
    try:
        evt = salt.utils.event.EventReturn(master_opts)
        evt.start()
    except TypeError as exc:
        if 'object' in str(exc):
            pytest.fail(f"'{exc}' TypeError should have not been raised")
    finally:
        if evt is not None:
            terminate_process(evt.pid, kill_children=True)

def test_filter_cluster_peer():
    if False:
        i = 10
        return i + 15
    assert salt.utils.event.EventReturn._filter({'__peer_id': 'foo', 'tag': 'salt/test', 'data': {'foo': 'bar'}}) is False

def test_filter_no_allow_or_deny():
    if False:
        print('Hello World!')
    assert salt.utils.event.EventReturn._filter({'tag': 'salt/test', 'data': {'foo': 'bar'}}) is True

def test_filter_not_allowed():
    if False:
        for i in range(10):
            print('nop')
    assert salt.utils.event.EventReturn._filter({'tag': 'salt/test', 'data': {'foo': 'bar'}}, allow=['foo/*']) is False

def test_filter_not_denied():
    if False:
        while True:
            i = 10
    assert salt.utils.event.EventReturn._filter({'tag': 'salt/test', 'data': {'foo': 'bar'}}, deny=['foo/*']) is True

def test_filter_allowed():
    if False:
        print('Hello World!')
    assert salt.utils.event.EventReturn._filter({'tag': 'salt/test', 'data': {'foo': 'bar'}}, allow=['salt/*']) is True

def test_filter_denied():
    if False:
        print('Hello World!')
    assert salt.utils.event.EventReturn._filter({'tag': 'salt/test', 'data': {'foo': 'bar'}}, deny=['salt/*']) is False

def test_filter_allowed_but_denied():
    if False:
        print('Hello World!')
    assert salt.utils.event.EventReturn._filter({'tag': 'salt/test', 'data': {'foo': 'bar'}}, allow=['salt/*'], deny=['salt/test']) is False