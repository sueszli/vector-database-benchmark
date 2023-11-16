import pytest
from watchdog.events import FileCreatedEvent, FileDeletedEvent, FileModifiedEvent, FileMovedEvent, DirCreatedEvent, DirDeletedEvent, DirMovedEvent
from maestral.sync import SyncEngine
from maestral.client import DropboxClient
from maestral.keyring import CredentialStorage
from maestral.config import remove_configuration

@pytest.fixture
def sync():
    if False:
        for i in range(10):
            print('nop')
    sync = SyncEngine(DropboxClient('test-config', CredentialStorage('test-config')))
    sync.dropbox_path = '/'
    yield sync
    remove_configuration('test-config')

def ipath(i):
    if False:
        i = 10
        return i + 15
    "Returns path names '/test 1', '/test 2', ..."
    return f'/test {i}'

def test_single_file_events(sync: SyncEngine) -> None:
    if False:
        return 10
    file_events = [FileModifiedEvent(ipath(1)), FileCreatedEvent(ipath(2)), FileDeletedEvent(ipath(3)), FileMovedEvent(ipath(4), ipath(5))]
    res = [FileModifiedEvent(ipath(1)), FileCreatedEvent(ipath(2)), FileDeletedEvent(ipath(3)), FileMovedEvent(ipath(4), ipath(5))]
    cleaned_events = sync._clean_local_events(file_events)
    assert cleaned_events == res

def test_single_path_cases(sync: SyncEngine) -> None:
    if False:
        return 10
    file_events = [FileCreatedEvent(ipath(1)), FileDeletedEvent(ipath(1)), FileDeletedEvent(ipath(2)), FileCreatedEvent(ipath(2)), FileCreatedEvent(ipath(3)), FileModifiedEvent(ipath(3))]
    res = [FileModifiedEvent(ipath(2)), FileCreatedEvent(ipath(3))]
    cleaned_events = sync._clean_local_events(file_events)
    assert cleaned_events == res

def test_move_events(sync: SyncEngine) -> None:
    if False:
        return 10
    file_events = [FileCreatedEvent(ipath(1)), FileMovedEvent(ipath(1), ipath(2)), FileMovedEvent(ipath(3), ipath(4)), FileDeletedEvent(ipath(4)), FileMovedEvent(ipath(5), ipath(6)), FileMovedEvent(ipath(6), ipath(5)), FileMovedEvent(ipath(7), ipath(8)), FileMovedEvent(ipath(8), ipath(9))]
    res = [FileCreatedEvent(ipath(2)), FileDeletedEvent(ipath(3)), FileModifiedEvent(ipath(5)), FileDeletedEvent(ipath(7)), FileCreatedEvent(ipath(9))]
    cleaned_events = sync._clean_local_events(file_events)
    assert cleaned_events == res

def test_gedit_save(sync: SyncEngine) -> None:
    if False:
        while True:
            i = 10
    file_events = [FileCreatedEvent('/.gedit-save-UR4EC0'), FileModifiedEvent('/.gedit-save-UR4EC0'), FileMovedEvent(ipath(1), ipath(1) + '~'), FileMovedEvent('/.gedit-save-UR4EC0', ipath(1))]
    res = [FileModifiedEvent(ipath(1)), FileCreatedEvent(ipath(1) + '~')]
    cleaned_events = sync._clean_local_events(file_events)
    assert cleaned_events == res

def test_macos_safe_save(sync: SyncEngine) -> None:
    if False:
        for i in range(10):
            print('nop')
    file_events = [FileMovedEvent(ipath(1), ipath(1) + '.sb-b78ef837-dLht38'), FileCreatedEvent(ipath(1)), FileDeletedEvent(ipath(1) + '.sb-b78ef837-dLht38')]
    res = [FileModifiedEvent(ipath(1))]
    cleaned_events = sync._clean_local_events(file_events)
    assert cleaned_events == res

def test_msoffice_created(sync: SyncEngine) -> None:
    if False:
        return 10
    file_events = [FileCreatedEvent(ipath(1)), FileDeletedEvent(ipath(1)), FileCreatedEvent(ipath(1)), FileCreatedEvent('/~$' + ipath(1))]
    res = [FileCreatedEvent(ipath(1)), FileCreatedEvent('/~$' + ipath(1))]
    cleaned_events = sync._clean_local_events(file_events)
    assert cleaned_events == res

def test_type_changes(sync: SyncEngine) -> None:
    if False:
        i = 10
        return i + 15
    file_events = [FileDeletedEvent(ipath(1)), DirCreatedEvent(ipath(1)), DirDeletedEvent(ipath(2)), FileCreatedEvent(ipath(2))]
    res = [FileDeletedEvent(ipath(1)), DirCreatedEvent(ipath(1)), DirDeletedEvent(ipath(2)), FileCreatedEvent(ipath(2))]
    cleaned_events = sync._clean_local_events(file_events)
    assert cleaned_events == res

def test_type_changes_difficult(sync: SyncEngine) -> None:
    if False:
        return 10
    file_events = [FileModifiedEvent(ipath(1)), FileDeletedEvent(ipath(1)), FileCreatedEvent(ipath(1)), FileDeletedEvent(ipath(1)), DirCreatedEvent(ipath(1)), FileModifiedEvent(ipath(2)), FileDeletedEvent(ipath(2)), FileCreatedEvent(ipath(2)), FileDeletedEvent(ipath(2)), DirCreatedEvent(ipath(2)), DirMovedEvent(ipath(2), ipath(3))]
    res = [FileDeletedEvent(ipath(1)), DirCreatedEvent(ipath(1)), FileDeletedEvent(ipath(2)), DirCreatedEvent(ipath(3))]
    cleaned_events = sync._clean_local_events(file_events)
    assert cleaned_events == res

def test_nested_events(sync: SyncEngine) -> None:
    if False:
        i = 10
        return i + 15
    file_events = [DirDeletedEvent(ipath(1)), FileDeletedEvent(ipath(1) + '/file1.txt'), FileDeletedEvent(ipath(1) + '/file2.txt'), DirDeletedEvent(ipath(1) + '/sub'), FileDeletedEvent(ipath(1) + '/sub/file3.txt'), DirMovedEvent(ipath(2), ipath(3)), FileMovedEvent(ipath(2) + '/file1.txt', ipath(3) + '/file1.txt'), FileMovedEvent(ipath(2) + '/file2.txt', ipath(3) + '/file2.txt'), DirMovedEvent(ipath(2) + '/sub', ipath(3) + '/sub'), FileMovedEvent(ipath(2) + '/sub/file3.txt', ipath(3) + '/sub/file3.txt')]
    res = [DirDeletedEvent(ipath(1)), DirMovedEvent(ipath(2), ipath(3))]
    cleaned_events = sync._clean_local_events(file_events)
    assert cleaned_events == res

@pytest.mark.benchmark(group='local-event-processing', min_time=0.1, max_time=5)
def test_performance(sync: SyncEngine, benchmark) -> None:
    if False:
        i = 10
        return i + 15
    file_events = [DirDeletedEvent(n * ipath(1)) for n in range(1, 5001)]
    file_events += [FileDeletedEvent(n * ipath(1) + '.txt') for n in range(1, 5001)]
    file_events += [DirMovedEvent(n * ipath(2), n * ipath(3)) for n in range(1, 5001)]
    file_events += [FileMovedEvent(n * ipath(2) + '.txt', n * ipath(3) + '.txt') for n in range(1, 5001)]
    file_events += [FileCreatedEvent(ipath(n)) for n in range(5, 5001)]
    res = [DirDeletedEvent(ipath(1)), FileDeletedEvent(ipath(1) + '.txt'), DirMovedEvent(ipath(2), ipath(3)), FileMovedEvent(ipath(2) + '.txt', ipath(3) + '.txt')]
    res += [FileCreatedEvent(ipath(n)) for n in range(5, 5001)]
    cleaned_events = benchmark(sync._clean_local_events, file_events)
    assert cleaned_events == res