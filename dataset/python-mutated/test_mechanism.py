import pytest
from sentry import eventstore
from sentry.event_manager import EventManager
from sentry.interfaces.exception import upgrade_legacy_mechanism

@pytest.fixture
def make_mechanism_snapshot(insta_snapshot):
    if False:
        return 10

    def inner(data):
        if False:
            while True:
                i = 10
        mgr = EventManager(data={'exception': {'values': [{'type': 'FooError', 'mechanism': data}]}})
        mgr.normalize()
        evt = eventstore.backend.create_event(data=mgr.get_data())
        mechanism = evt.interfaces['exception'].values[0].mechanism
        insta_snapshot({'errors': evt.data.get('errors'), 'to_json': mechanism.to_json(), 'tags': sorted(mechanism.iter_tags())})
    return inner

def test_empty_mechanism(make_mechanism_snapshot):
    if False:
        return 10
    data = {'type': 'generic'}
    make_mechanism_snapshot(data)

def test_tag(make_mechanism_snapshot):
    if False:
        for i in range(10):
            print('nop')
    data = {'type': 'generic'}
    make_mechanism_snapshot(data)

def test_tag_with_handled(make_mechanism_snapshot):
    if False:
        for i in range(10):
            print('nop')
    data = {'type': 'generic', 'handled': False}
    make_mechanism_snapshot(data)

def test_data(make_mechanism_snapshot):
    if False:
        print('Hello World!')
    data = {'type': 'generic', 'data': {'relevant_address': '0x1'}}
    make_mechanism_snapshot(data)

def test_empty_data(make_mechanism_snapshot):
    if False:
        print('Hello World!')
    data = {'type': 'generic', 'data': {}}
    make_mechanism_snapshot(data)

def test_min_mach_meta(make_mechanism_snapshot):
    if False:
        i = 10
        return i + 15
    input = {'type': 'generic', 'meta': {'mach_exception': {'exception': 10, 'code': 0, 'subcode': 0}}}
    make_mechanism_snapshot(input)

def test_full_mach_meta(make_mechanism_snapshot):
    if False:
        for i in range(10):
            print('nop')
    data = {'type': 'generic', 'meta': {'mach_exception': {'exception': 10, 'code': 0, 'subcode': 0, 'name': 'EXC_CRASH'}}}
    make_mechanism_snapshot(data)

def test_min_signal_meta(make_mechanism_snapshot):
    if False:
        while True:
            i = 10
    data = {'type': 'generic', 'meta': {'signal': {'number': 10, 'code': 0}}}
    make_mechanism_snapshot(data)

def test_full_signal_meta(make_mechanism_snapshot):
    if False:
        i = 10
        return i + 15
    data = {'type': 'generic', 'meta': {'signal': {'number': 10, 'code': 0, 'name': 'SIGBUS', 'code_name': 'BUS_NOOP'}}}
    make_mechanism_snapshot(data)

def test_min_errno_meta(make_mechanism_snapshot):
    if False:
        i = 10
        return i + 15
    data = {'type': 'generic', 'meta': {'errno': {'number': 2}}}
    make_mechanism_snapshot(data)

def test_full_errno_meta(make_mechanism_snapshot):
    if False:
        return 10
    data = {'type': 'generic', 'meta': {'errno': {'number': 2, 'name': 'ENOENT'}}}
    make_mechanism_snapshot(data)

def test_upgrade():
    if False:
        return 10
    data = {'posix_signal': {'name': 'SIGSEGV', 'code_name': 'SEGV_NOOP', 'signal': 11, 'code': 0}, 'relevant_address': '0x1', 'mach_exception': {'exception': 1, 'exception_name': 'EXC_BAD_ACCESS', 'subcode': 8, 'code': 1}}
    assert upgrade_legacy_mechanism(data) == {'type': 'generic', 'data': {'relevant_address': '0x1'}, 'meta': {'mach_exception': {'exception': 1, 'subcode': 8, 'code': 1, 'name': 'EXC_BAD_ACCESS'}, 'signal': {'number': 11, 'code': 0, 'name': 'SIGSEGV', 'code_name': 'SEGV_NOOP'}}}