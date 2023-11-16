import time
from collections import namedtuple
import pytest
import salt.modules.ps
import salt.modules.ps as ps
import salt.utils.data
from salt.exceptions import SaltInvocationError
from tests.support.mock import MagicMock, Mock, call, patch
psutil = pytest.importorskip('salt.utils.psutil_compat')

@pytest.fixture
def sample_process():
    if False:
        i = 10
        return i + 15
    status = b'fnord'
    extra_data = {'utime': '42', 'stime': '42', 'children_utime': '42', 'children_stime': '42', 'ttynr': '42', 'cpu_time': '42', 'blkio_ticks': '99', 'ppid': '99', 'cpu_num': '9999999'}
    important_data = {'name': b'blerp', 'status': status, 'create_time': '393829200'}
    important_data.update(extra_data)
    patch_stat_file = patch('psutil._psplatform.Process._parse_stat_file', return_value=important_data, create=True)
    patch_exe = patch('psutil._psplatform.Process.exe', return_value=important_data['name'].decode(), create=True)
    patch_oneshot = patch('psutil._psplatform.Process.oneshot', return_value={1: important_data['status'].decode(), 9: float(important_data['create_time']), 14: float(important_data['create_time']), 15: float(important_data['create_time']), 16: float(important_data['create_time']), 17: float(important_data['create_time']), 24: important_data['name'].decode()}, create=True)
    patch_kinfo = patch('psutil._psplatform.Process._get_kinfo_proc', return_value={9: important_data['status'].decode(), 8: float(important_data['create_time']), 10: important_data['name'].decode()}, create=True)
    patch_status = patch('psutil._psplatform.Process.status', return_value=status.decode())
    patch_create_time = patch('psutil._psplatform.Process.create_time', return_value=393829200)
    with patch_stat_file, patch_status, patch_create_time, patch_exe, patch_oneshot, patch_kinfo:
        proc = psutil.Process(pid=42)
        proc.info = proc.as_dict(('name', 'status'))
        yield proc

def test__status_when_process_is_found_with_matching_status_then_proc_info_should_be_returned(sample_process):
    if False:
        i = 10
        return i + 15
    expected_result = [{'pid': 42, 'name': 'blerp'}]
    proc = sample_process
    with patch('salt.utils.psutil_compat.process_iter', autospec=True, return_value=[proc]):
        actual_result = salt.modules.ps.status(status='fnord')
        assert actual_result == expected_result

def test__status_when_no_matching_processes_then_no_results_should_be_returned():
    if False:
        return 10
    expected_result = []
    with patch('salt.utils.psutil_compat.process_iter', autospec=True, return_value=[MagicMock(info={'status': 'foo', 'blerp': 'whatever'})]):
        actual_result = salt.modules.ps.status(status='fnord')
        assert actual_result == expected_result

def test__status_when_some_matching_processes_then_only_correct_info_should_be_returned(sample_process):
    if False:
        return 10
    expected_result = [{'name': 'blerp', 'pid': 42}]
    with patch('salt.utils.psutil_compat.process_iter', autospec=True, return_value=[sample_process, MagicMock(info={'status': 'foo', 'name': 'wherever', 'pid': 9998}), MagicMock(info={'status': 'bar', 'name': 'whenever', 'pid': 9997})]):
        actual_result = salt.modules.ps.status(status='fnord')
        assert actual_result == expected_result
HAS_PSUTIL_VERSION = False
PSUTIL2 = psutil.version_info >= (2, 0)
STUB_CPU_TIMES = namedtuple('cputimes', 'user nice system idle')(1, 2, 3, 4)
STUB_VIRT_MEM = namedtuple('vmem', 'total available percent used free')(1000, 500, 50, 500, 500)
STUB_SWAP_MEM = namedtuple('swap', 'total used free percent sin sout')(1000, 500, 500, 50, 0, 0)
STUB_PHY_MEM_USAGE = namedtuple('usage', 'total used free percent')(1000, 500, 500, 50)
STUB_DISK_PARTITION = namedtuple('partition', 'device mountpoint fstype, opts')('/dev/disk0s2', '/', 'hfs', 'rw,local,rootfs,dovolfs,journaled,multilabel')
STUB_DISK_USAGE = namedtuple('usage', 'total used free percent')(1000, 500, 500, 50)
STUB_NETWORK_IO = namedtuple('iostat', 'bytes_sent, bytes_recv, packets_sent, packets_recv, errin errout dropin dropout')(1000, 2000, 500, 600, 1, 2, 3, 4)
STUB_DISK_IO = namedtuple('iostat', 'read_count, write_count, read_bytes, write_bytes, read_time, write_time')(1000, 2000, 500, 600, 2000, 3000)

@pytest.fixture(scope='module')
def stub_user():
    if False:
        for i in range(10):
            print('nop')
    return namedtuple('user', 'name, terminal, host, started')('bdobbs', 'ttys000', 'localhost', 0.0)
if psutil.version_info >= (0, 6, 0):
    HAS_PSUTIL_VERSION = True
STUB_PID_LIST = [0, 1, 2, 3]
try:
    import utmp
    HAS_UTMP = True
except ImportError:
    HAS_UTMP = False

def _get_proc_name(proc):
    if False:
        for i in range(10):
            print('nop')
    return proc.name() if PSUTIL2 else proc.name

def _get_proc_pid(proc):
    if False:
        while True:
            i = 10
    return proc.pid

class DummyProcess:
    """
    Dummy class to emulate psutil.Process. This ensures that _any_ string
    values used for any of the options passed in are converted to str types on
    both Python 2 and Python 3.
    """

    def __init__(self, cmdline=None, create_time=None, name=None, status=None, username=None, pid=None):
        if False:
            while True:
                i = 10
        self._cmdline = salt.utils.data.decode(cmdline if cmdline is not None else [], to_str=True)
        self._create_time = salt.utils.data.decode(create_time if create_time is not None else time.time(), to_str=True)
        self._name = salt.utils.data.decode(name if name is not None else [], to_str=True)
        self._status = salt.utils.data.decode(status, to_str=True)
        self._username = salt.utils.data.decode(username, to_str=True)
        self._pid = salt.utils.data.decode(pid if pid is not None else 12345, to_str=True)

    def cmdline(self):
        if False:
            print('Hello World!')
        return self._cmdline

    def create_time(self):
        if False:
            return 10
        return self._create_time

    def name(self):
        if False:
            for i in range(10):
                print('nop')
        return self._name

    def status(self):
        if False:
            i = 10
            return i + 15
        return self._status

    def username(self):
        if False:
            while True:
                i = 10
        return self._username

    def pid(self):
        if False:
            print('Hello World!')
        return self._pid

@pytest.fixture
def mocked_proc():
    if False:
        i = 10
        return i + 15
    mocked_proc = MagicMock('salt.utils.psutil_compat.Process')
    if PSUTIL2:
        mocked_proc.name = Mock(return_value='test_mock_proc')
        mocked_proc.pid = Mock(return_value=9999999999)
    else:
        mocked_proc.name = 'test_mock_proc'
        mocked_proc.pid = 9999999999
    with patch('salt.utils.psutil_compat.Process.send_signal'), patch('salt.utils.psutil_compat.process_iter', MagicMock(return_value=[mocked_proc])):
        yield mocked_proc

@pytest.mark.skipif(not ps.PSUTIL2, reason='Only run for psutil 2.x')
def test__get_proc_cmdline():
    if False:
        while True:
            i = 10
    cmdline = ['echo', 'питон']
    ret = ps._get_proc_cmdline(DummyProcess(cmdline=cmdline))
    assert ret == cmdline, ret

def test_get_pid_list():
    if False:
        for i in range(10):
            print('nop')
    with patch('salt.utils.psutil_compat.pids', MagicMock(return_value=STUB_PID_LIST)):
        assert STUB_PID_LIST == ps.get_pid_list()

def test_kill_pid():
    if False:
        while True:
            i = 10
    with patch('salt.utils.psutil_compat.Process') as send_signal_mock:
        ps.kill_pid(0, signal=999)
        assert send_signal_mock.call_args == call(0)

def test_pkill(mocked_proc):
    if False:
        while True:
            i = 10
    mocked_proc.send_signal = MagicMock()
    test_signal = 1234
    ps.pkill(_get_proc_name(mocked_proc), signal=test_signal)
    assert mocked_proc.send_signal.call_args == call(test_signal)

def test_pgrep(mocked_proc):
    if False:
        print('Hello World!')
    with patch('salt.utils.psutil_compat.process_iter', MagicMock(return_value=[mocked_proc])):
        assert mocked_proc.pid in (ps.pgrep(_get_proc_name(mocked_proc)) or [])

def test_pgrep_regex(mocked_proc):
    if False:
        for i in range(10):
            print('nop')
    with patch('salt.utils.psutil_compat.process_iter', MagicMock(return_value=[mocked_proc])):
        assert mocked_proc.pid in (ps.pgrep('t.st_[a-z]+_proc', pattern_is_regex=True) or [])

def test_cpu_percent():
    if False:
        for i in range(10):
            print('nop')
    with patch('salt.utils.psutil_compat.cpu_percent', MagicMock(return_value=1)):
        assert ps.cpu_percent() == 1

def test_cpu_times():
    if False:
        for i in range(10):
            print('nop')
    with patch('salt.utils.psutil_compat.cpu_times', MagicMock(return_value=STUB_CPU_TIMES)):
        assert {'idle': 4, 'nice': 2, 'system': 3, 'user': 1} == ps.cpu_times()

@pytest.mark.skipif(HAS_PSUTIL_VERSION is False, reason='psutil 0.6.0 or greater is required for this test')
def test_virtual_memory():
    if False:
        for i in range(10):
            print('nop')
    with patch('salt.utils.psutil_compat.virtual_memory', MagicMock(return_value=STUB_VIRT_MEM)):
        assert {'used': 500, 'total': 1000, 'available': 500, 'percent': 50, 'free': 500} == ps.virtual_memory()

@pytest.mark.skipif(HAS_PSUTIL_VERSION is False, reason='psutil 0.6.0 or greater is required for this test')
def test_swap_memory():
    if False:
        i = 10
        return i + 15
    with patch('salt.utils.psutil_compat.swap_memory', MagicMock(return_value=STUB_SWAP_MEM)):
        assert {'used': 500, 'total': 1000, 'percent': 50, 'free': 500, 'sin': 0, 'sout': 0} == ps.swap_memory()

def test_disk_partitions():
    if False:
        return 10
    with patch('salt.utils.psutil_compat.disk_partitions', MagicMock(return_value=[STUB_DISK_PARTITION])):
        assert {'device': '/dev/disk0s2', 'mountpoint': '/', 'opts': 'rw,local,rootfs,dovolfs,journaled,multilabel', 'fstype': 'hfs'} == ps.disk_partitions()[0]

def test_disk_usage():
    if False:
        print('Hello World!')
    with patch('salt.utils.psutil_compat.disk_usage', MagicMock(return_value=STUB_DISK_USAGE)):
        assert {'used': 500, 'total': 1000, 'percent': 50, 'free': 500} == ps.disk_usage('DUMMY_PATH')

def test_disk_partition_usage():
    if False:
        print('Hello World!')
    with patch('salt.utils.psutil_compat.disk_partitions', MagicMock(return_value=[STUB_DISK_PARTITION])):
        assert {'device': '/dev/disk0s2', 'mountpoint': '/', 'opts': 'rw,local,rootfs,dovolfs,journaled,multilabel', 'fstype': 'hfs'} == ps.disk_partitions()[0]

def test_network_io_counters():
    if False:
        i = 10
        return i + 15
    with patch('salt.utils.psutil_compat.net_io_counters', MagicMock(return_value=STUB_NETWORK_IO)):
        assert {'packets_sent': 500, 'packets_recv': 600, 'bytes_recv': 2000, 'dropout': 4, 'bytes_sent': 1000, 'errout': 2, 'errin': 1, 'dropin': 3} == ps.network_io_counters()

def test_disk_io_counters():
    if False:
        print('Hello World!')
    with patch('salt.utils.psutil_compat.disk_io_counters', MagicMock(return_value=STUB_DISK_IO)):
        assert {'read_time': 2000, 'write_bytes': 600, 'read_bytes': 500, 'write_time': 3000, 'read_count': 1000, 'write_count': 2000} == ps.disk_io_counters()

def test_get_users(stub_user):
    if False:
        i = 10
        return i + 15
    with patch('salt.utils.psutil_compat.users', MagicMock(return_value=[stub_user])):
        assert {'terminal': 'ttys000', 'started': 0.0, 'host': 'localhost', 'name': 'bdobbs'} == ps.get_users()[0]

def test_top():
    if False:
        for i in range(10):
            print('nop')
    '\n    See the following issue:\n\n    https://github.com/saltstack/salt/issues/56942\n    '
    result = ps.top(num_processes=1, interval=0)
    assert len(result) == 1

def test_top_zombie_process():
    if False:
        i = 10
        return i + 15
    pids = psutil.pids()[:3]
    processes = [psutil.Process(pid) for pid in pids]

    def raise_exception():
        if False:
            return 10
        raise psutil.ZombieProcess(processes[1].pid)
    processes[1].cpu_times = raise_exception
    with patch('salt.utils.psutil_compat.pids', return_value=pids):
        with patch('salt.utils.psutil_compat.Process', side_effect=processes):
            result = ps.top(num_processes=1, interval=0)
            assert len(result) == 1

def test_status_when_no_status_is_provided_then_raise_invocation_error():
    if False:
        return 10
    with pytest.raises(SaltInvocationError):
        actual_result = salt.modules.ps.status(status='')

@pytest.mark.parametrize('exc_type', (salt.utils.psutil_compat.AccessDenied(pid='9999', name='whatever'), salt.utils.psutil_compat.NoSuchProcess(pid='42')))
def test_status_when_access_denied_from_psutil_it_should_CommandExecutionError(exc_type):
    if False:
        i = 10
        return i + 15
    with patch('salt.utils.psutil_compat.process_iter', autospec=True, side_effect=exc_type):
        with pytest.raises(salt.exceptions.CommandExecutionError, match='Psutil did not return a list of processes'):
            actual_result = salt.modules.ps.status(status='fnord')

def test_status_when_no_filter_is_provided_then_raise_invocation_error():
    if False:
        while True:
            i = 10
    with pytest.raises(SaltInvocationError) as invoc_issue:
        actual_result = salt.modules.ps.status(status='')

def test_status_when_access_denied_from_psutil_then_raise_exception():
    if False:
        i = 10
        return i + 15
    with patch('salt.utils.psutil_compat.process_iter', autospec=True, return_value=salt.utils.psutil_compat.AccessDenied(pid='9999', name='whatever')):
        with pytest.raises(Exception) as general_issue:
            actual_result = salt.modules.ps.status(status='fnord')