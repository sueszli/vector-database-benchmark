"""Sun OS Solaris platform implementation."""
import errno
import functools
import os
import socket
import subprocess
import sys
from collections import namedtuple
from socket import AF_INET
from . import _common
from . import _psposix
from . import _psutil_posix as cext_posix
from . import _psutil_sunos as cext
from ._common import AF_INET6
from ._common import AccessDenied
from ._common import NoSuchProcess
from ._common import ZombieProcess
from ._common import debug
from ._common import get_procfs_path
from ._common import isfile_strict
from ._common import memoize_when_activated
from ._common import sockfam_to_enum
from ._common import socktype_to_enum
from ._common import usage_percent
from ._compat import PY3
from ._compat import FileNotFoundError
from ._compat import PermissionError
from ._compat import ProcessLookupError
from ._compat import b
__extra__all__ = ['CONN_IDLE', 'CONN_BOUND', 'PROCFS_PATH']
PAGE_SIZE = cext_posix.getpagesize()
AF_LINK = cext_posix.AF_LINK
IS_64_BIT = sys.maxsize > 2 ** 32
CONN_IDLE = 'IDLE'
CONN_BOUND = 'BOUND'
PROC_STATUSES = {cext.SSLEEP: _common.STATUS_SLEEPING, cext.SRUN: _common.STATUS_RUNNING, cext.SZOMB: _common.STATUS_ZOMBIE, cext.SSTOP: _common.STATUS_STOPPED, cext.SIDL: _common.STATUS_IDLE, cext.SONPROC: _common.STATUS_RUNNING, cext.SWAIT: _common.STATUS_WAITING}
TCP_STATUSES = {cext.TCPS_ESTABLISHED: _common.CONN_ESTABLISHED, cext.TCPS_SYN_SENT: _common.CONN_SYN_SENT, cext.TCPS_SYN_RCVD: _common.CONN_SYN_RECV, cext.TCPS_FIN_WAIT_1: _common.CONN_FIN_WAIT1, cext.TCPS_FIN_WAIT_2: _common.CONN_FIN_WAIT2, cext.TCPS_TIME_WAIT: _common.CONN_TIME_WAIT, cext.TCPS_CLOSED: _common.CONN_CLOSE, cext.TCPS_CLOSE_WAIT: _common.CONN_CLOSE_WAIT, cext.TCPS_LAST_ACK: _common.CONN_LAST_ACK, cext.TCPS_LISTEN: _common.CONN_LISTEN, cext.TCPS_CLOSING: _common.CONN_CLOSING, cext.PSUTIL_CONN_NONE: _common.CONN_NONE, cext.TCPS_IDLE: CONN_IDLE, cext.TCPS_BOUND: CONN_BOUND}
proc_info_map = dict(ppid=0, rss=1, vms=2, create_time=3, nice=4, num_threads=5, status=6, ttynr=7, uid=8, euid=9, gid=10, egid=11)
scputimes = namedtuple('scputimes', ['user', 'system', 'idle', 'iowait'])
pcputimes = namedtuple('pcputimes', ['user', 'system', 'children_user', 'children_system'])
svmem = namedtuple('svmem', ['total', 'available', 'percent', 'used', 'free'])
pmem = namedtuple('pmem', ['rss', 'vms'])
pfullmem = pmem
pmmap_grouped = namedtuple('pmmap_grouped', ['path', 'rss', 'anonymous', 'locked'])
pmmap_ext = namedtuple('pmmap_ext', 'addr perms ' + ' '.join(pmmap_grouped._fields))

def virtual_memory():
    if False:
        print('Hello World!')
    'Report virtual memory metrics.'
    total = os.sysconf('SC_PHYS_PAGES') * PAGE_SIZE
    free = avail = os.sysconf('SC_AVPHYS_PAGES') * PAGE_SIZE
    used = total - free
    percent = usage_percent(used, total, round_=1)
    return svmem(total, avail, percent, used, free)

def swap_memory():
    if False:
        for i in range(10):
            print('nop')
    'Report swap memory metrics.'
    (sin, sout) = cext.swap_mem()
    p = subprocess.Popen(['/usr/bin/env', 'PATH=/usr/sbin:/sbin:%s' % os.environ['PATH'], 'swap', '-l'], stdout=subprocess.PIPE)
    (stdout, _) = p.communicate()
    if PY3:
        stdout = stdout.decode(sys.stdout.encoding)
    if p.returncode != 0:
        raise RuntimeError("'swap -l' failed (retcode=%s)" % p.returncode)
    lines = stdout.strip().split('\n')[1:]
    if not lines:
        msg = 'no swap device(s) configured'
        raise RuntimeError(msg)
    total = free = 0
    for line in lines:
        line = line.split()
        (t, f) = line[3:5]
        total += int(int(t) * 512)
        free += int(int(f) * 512)
    used = total - free
    percent = usage_percent(used, total, round_=1)
    return _common.sswap(total, used, free, percent, sin * PAGE_SIZE, sout * PAGE_SIZE)

def cpu_times():
    if False:
        return 10
    'Return system-wide CPU times as a named tuple.'
    ret = cext.per_cpu_times()
    return scputimes(*[sum(x) for x in zip(*ret)])

def per_cpu_times():
    if False:
        for i in range(10):
            print('nop')
    'Return system per-CPU times as a list of named tuples.'
    ret = cext.per_cpu_times()
    return [scputimes(*x) for x in ret]

def cpu_count_logical():
    if False:
        for i in range(10):
            print('nop')
    'Return the number of logical CPUs in the system.'
    try:
        return os.sysconf('SC_NPROCESSORS_ONLN')
    except ValueError:
        return None

def cpu_count_cores():
    if False:
        while True:
            i = 10
    'Return the number of CPU cores in the system.'
    return cext.cpu_count_cores()

def cpu_stats():
    if False:
        while True:
            i = 10
    'Return various CPU stats as a named tuple.'
    (ctx_switches, interrupts, syscalls, traps) = cext.cpu_stats()
    soft_interrupts = 0
    return _common.scpustats(ctx_switches, interrupts, soft_interrupts, syscalls)
disk_io_counters = cext.disk_io_counters
disk_usage = _psposix.disk_usage

def disk_partitions(all=False):
    if False:
        for i in range(10):
            print('nop')
    'Return system disk partitions.'
    retlist = []
    partitions = cext.disk_partitions()
    for partition in partitions:
        (device, mountpoint, fstype, opts) = partition
        if device == 'none':
            device = ''
        if not all:
            try:
                if not disk_usage(mountpoint).total:
                    continue
            except OSError as err:
                debug('skipping %r: %s' % (mountpoint, err))
                continue
        maxfile = maxpath = None
        ntuple = _common.sdiskpart(device, mountpoint, fstype, opts, maxfile, maxpath)
        retlist.append(ntuple)
    return retlist
net_io_counters = cext.net_io_counters
net_if_addrs = cext_posix.net_if_addrs

def net_connections(kind, _pid=-1):
    if False:
        i = 10
        return i + 15
    'Return socket connections.  If pid == -1 return system-wide\n    connections (as opposed to connections opened by one process only).\n    Only INET sockets are returned (UNIX are not).\n    '
    cmap = _common.conn_tmap.copy()
    if _pid == -1:
        cmap.pop('unix', 0)
    if kind not in cmap:
        raise ValueError('invalid %r kind argument; choose between %s' % (kind, ', '.join([repr(x) for x in cmap])))
    (families, types) = _common.conn_tmap[kind]
    rawlist = cext.net_connections(_pid)
    ret = set()
    for item in rawlist:
        (fd, fam, type_, laddr, raddr, status, pid) = item
        if fam not in families:
            continue
        if type_ not in types:
            continue
        if fam in (AF_INET, AF_INET6):
            if laddr:
                laddr = _common.addr(*laddr)
            if raddr:
                raddr = _common.addr(*raddr)
        status = TCP_STATUSES[status]
        fam = sockfam_to_enum(fam)
        type_ = socktype_to_enum(type_)
        if _pid == -1:
            nt = _common.sconn(fd, fam, type_, laddr, raddr, status, pid)
        else:
            nt = _common.pconn(fd, fam, type_, laddr, raddr, status)
        ret.add(nt)
    return list(ret)

def net_if_stats():
    if False:
        return 10
    'Get NIC stats (isup, duplex, speed, mtu).'
    ret = cext.net_if_stats()
    for (name, items) in ret.items():
        (isup, duplex, speed, mtu) = items
        if hasattr(_common, 'NicDuplex'):
            duplex = _common.NicDuplex(duplex)
        ret[name] = _common.snicstats(isup, duplex, speed, mtu, '')
    return ret

def boot_time():
    if False:
        for i in range(10):
            print('nop')
    'The system boot time expressed in seconds since the epoch.'
    return cext.boot_time()

def users():
    if False:
        for i in range(10):
            print('nop')
    'Return currently connected users as a list of namedtuples.'
    retlist = []
    rawlist = cext.users()
    localhost = (':0.0', ':0')
    for item in rawlist:
        (user, tty, hostname, tstamp, user_process, pid) = item
        if not user_process:
            continue
        if hostname in localhost:
            hostname = 'localhost'
        nt = _common.suser(user, tty, hostname, tstamp, pid)
        retlist.append(nt)
    return retlist

def pids():
    if False:
        while True:
            i = 10
    'Returns a list of PIDs currently running on the system.'
    return [int(x) for x in os.listdir(b(get_procfs_path())) if x.isdigit()]

def pid_exists(pid):
    if False:
        print('Hello World!')
    'Check for the existence of a unix pid.'
    return _psposix.pid_exists(pid)

def wrap_exceptions(fun):
    if False:
        print('Hello World!')
    'Call callable into a try/except clause and translate ENOENT,\n    EACCES and EPERM in NoSuchProcess or AccessDenied exceptions.\n    '

    @functools.wraps(fun)
    def wrapper(self, *args, **kwargs):
        if False:
            print('Hello World!')
        try:
            return fun(self, *args, **kwargs)
        except (FileNotFoundError, ProcessLookupError):
            if not pid_exists(self.pid):
                raise NoSuchProcess(self.pid, self._name)
            else:
                raise ZombieProcess(self.pid, self._name, self._ppid)
        except PermissionError:
            raise AccessDenied(self.pid, self._name)
        except OSError:
            if self.pid == 0:
                if 0 in pids():
                    raise AccessDenied(self.pid, self._name)
                else:
                    raise
            raise
    return wrapper

class Process:
    """Wrapper class around underlying C implementation."""
    __slots__ = ['pid', '_name', '_ppid', '_procfs_path', '_cache']

    def __init__(self, pid):
        if False:
            i = 10
            return i + 15
        self.pid = pid
        self._name = None
        self._ppid = None
        self._procfs_path = get_procfs_path()

    def _assert_alive(self):
        if False:
            print('Hello World!')
        'Raise NSP if the process disappeared on us.'
        os.stat('%s/%s' % (self._procfs_path, self.pid))

    def oneshot_enter(self):
        if False:
            for i in range(10):
                print('nop')
        self._proc_name_and_args.cache_activate(self)
        self._proc_basic_info.cache_activate(self)
        self._proc_cred.cache_activate(self)

    def oneshot_exit(self):
        if False:
            i = 10
            return i + 15
        self._proc_name_and_args.cache_deactivate(self)
        self._proc_basic_info.cache_deactivate(self)
        self._proc_cred.cache_deactivate(self)

    @wrap_exceptions
    @memoize_when_activated
    def _proc_name_and_args(self):
        if False:
            i = 10
            return i + 15
        return cext.proc_name_and_args(self.pid, self._procfs_path)

    @wrap_exceptions
    @memoize_when_activated
    def _proc_basic_info(self):
        if False:
            print('Hello World!')
        if self.pid == 0 and (not os.path.exists('%s/%s/psinfo' % (self._procfs_path, self.pid))):
            raise AccessDenied(self.pid)
        ret = cext.proc_basic_info(self.pid, self._procfs_path)
        assert len(ret) == len(proc_info_map)
        return ret

    @wrap_exceptions
    @memoize_when_activated
    def _proc_cred(self):
        if False:
            return 10
        return cext.proc_cred(self.pid, self._procfs_path)

    @wrap_exceptions
    def name(self):
        if False:
            while True:
                i = 10
        return self._proc_name_and_args()[0]

    @wrap_exceptions
    def exe(self):
        if False:
            return 10
        try:
            return os.readlink('%s/%s/path/a.out' % (self._procfs_path, self.pid))
        except OSError:
            pass
        self.cmdline()
        return ''

    @wrap_exceptions
    def cmdline(self):
        if False:
            return 10
        return self._proc_name_and_args()[1].split(' ')

    @wrap_exceptions
    def environ(self):
        if False:
            i = 10
            return i + 15
        return cext.proc_environ(self.pid, self._procfs_path)

    @wrap_exceptions
    def create_time(self):
        if False:
            i = 10
            return i + 15
        return self._proc_basic_info()[proc_info_map['create_time']]

    @wrap_exceptions
    def num_threads(self):
        if False:
            for i in range(10):
                print('nop')
        return self._proc_basic_info()[proc_info_map['num_threads']]

    @wrap_exceptions
    def nice_get(self):
        if False:
            i = 10
            return i + 15
        return self._proc_basic_info()[proc_info_map['nice']]

    @wrap_exceptions
    def nice_set(self, value):
        if False:
            for i in range(10):
                print('nop')
        if self.pid in (2, 3):
            raise AccessDenied(self.pid, self._name)
        return cext_posix.setpriority(self.pid, value)

    @wrap_exceptions
    def ppid(self):
        if False:
            print('Hello World!')
        self._ppid = self._proc_basic_info()[proc_info_map['ppid']]
        return self._ppid

    @wrap_exceptions
    def uids(self):
        if False:
            i = 10
            return i + 15
        try:
            (real, effective, saved, _, _, _) = self._proc_cred()
        except AccessDenied:
            real = self._proc_basic_info()[proc_info_map['uid']]
            effective = self._proc_basic_info()[proc_info_map['euid']]
            saved = None
        return _common.puids(real, effective, saved)

    @wrap_exceptions
    def gids(self):
        if False:
            print('Hello World!')
        try:
            (_, _, _, real, effective, saved) = self._proc_cred()
        except AccessDenied:
            real = self._proc_basic_info()[proc_info_map['gid']]
            effective = self._proc_basic_info()[proc_info_map['egid']]
            saved = None
        return _common.puids(real, effective, saved)

    @wrap_exceptions
    def cpu_times(self):
        if False:
            print('Hello World!')
        try:
            times = cext.proc_cpu_times(self.pid, self._procfs_path)
        except OSError as err:
            if err.errno == errno.EOVERFLOW and (not IS_64_BIT):
                times = (0.0, 0.0, 0.0, 0.0)
            else:
                raise
        return _common.pcputimes(*times)

    @wrap_exceptions
    def cpu_num(self):
        if False:
            return 10
        return cext.proc_cpu_num(self.pid, self._procfs_path)

    @wrap_exceptions
    def terminal(self):
        if False:
            while True:
                i = 10
        procfs_path = self._procfs_path
        hit_enoent = False
        tty = wrap_exceptions(self._proc_basic_info()[proc_info_map['ttynr']])
        if tty != cext.PRNODEV:
            for x in (0, 1, 2, 255):
                try:
                    return os.readlink('%s/%d/path/%d' % (procfs_path, self.pid, x))
                except FileNotFoundError:
                    hit_enoent = True
                    continue
        if hit_enoent:
            self._assert_alive()

    @wrap_exceptions
    def cwd(self):
        if False:
            i = 10
            return i + 15
        procfs_path = self._procfs_path
        try:
            return os.readlink('%s/%s/path/cwd' % (procfs_path, self.pid))
        except FileNotFoundError:
            os.stat('%s/%s' % (procfs_path, self.pid))
            return ''

    @wrap_exceptions
    def memory_info(self):
        if False:
            print('Hello World!')
        ret = self._proc_basic_info()
        rss = ret[proc_info_map['rss']] * 1024
        vms = ret[proc_info_map['vms']] * 1024
        return pmem(rss, vms)
    memory_full_info = memory_info

    @wrap_exceptions
    def status(self):
        if False:
            print('Hello World!')
        code = self._proc_basic_info()[proc_info_map['status']]
        return PROC_STATUSES.get(code, '?')

    @wrap_exceptions
    def threads(self):
        if False:
            return 10
        procfs_path = self._procfs_path
        ret = []
        tids = os.listdir('%s/%d/lwp' % (procfs_path, self.pid))
        hit_enoent = False
        for tid in tids:
            tid = int(tid)
            try:
                (utime, stime) = cext.query_process_thread(self.pid, tid, procfs_path)
            except EnvironmentError as err:
                if err.errno == errno.EOVERFLOW and (not IS_64_BIT):
                    continue
                if err.errno == errno.ENOENT:
                    hit_enoent = True
                    continue
                raise
            else:
                nt = _common.pthread(tid, utime, stime)
                ret.append(nt)
        if hit_enoent:
            self._assert_alive()
        return ret

    @wrap_exceptions
    def open_files(self):
        if False:
            print('Hello World!')
        retlist = []
        hit_enoent = False
        procfs_path = self._procfs_path
        pathdir = '%s/%d/path' % (procfs_path, self.pid)
        for fd in os.listdir('%s/%d/fd' % (procfs_path, self.pid)):
            path = os.path.join(pathdir, fd)
            if os.path.islink(path):
                try:
                    file = os.readlink(path)
                except FileNotFoundError:
                    hit_enoent = True
                    continue
                else:
                    if isfile_strict(file):
                        retlist.append(_common.popenfile(file, int(fd)))
        if hit_enoent:
            self._assert_alive()
        return retlist

    def _get_unix_sockets(self, pid):
        if False:
            return 10
        "Get UNIX sockets used by process by parsing 'pfiles' output."
        cmd = ['pfiles', str(pid)]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (stdout, stderr) = p.communicate()
        if PY3:
            (stdout, stderr) = (x.decode(sys.stdout.encoding) for x in (stdout, stderr))
        if p.returncode != 0:
            if 'permission denied' in stderr.lower():
                raise AccessDenied(self.pid, self._name)
            if 'no such process' in stderr.lower():
                raise NoSuchProcess(self.pid, self._name)
            raise RuntimeError('%r command error\n%s' % (cmd, stderr))
        lines = stdout.split('\n')[2:]
        for (i, line) in enumerate(lines):
            line = line.lstrip()
            if line.startswith('sockname: AF_UNIX'):
                path = line.split(' ', 2)[2]
                type = lines[i - 2].strip()
                if type == 'SOCK_STREAM':
                    type = socket.SOCK_STREAM
                elif type == 'SOCK_DGRAM':
                    type = socket.SOCK_DGRAM
                else:
                    type = -1
                yield (-1, socket.AF_UNIX, type, path, '', _common.CONN_NONE)

    @wrap_exceptions
    def connections(self, kind='inet'):
        if False:
            for i in range(10):
                print('nop')
        ret = net_connections(kind, _pid=self.pid)
        if not ret:
            os.stat('%s/%s' % (self._procfs_path, self.pid))
        if kind in ('all', 'unix'):
            ret.extend([_common.pconn(*conn) for conn in self._get_unix_sockets(self.pid)])
        return ret
    nt_mmap_grouped = namedtuple('mmap', 'path rss anon locked')
    nt_mmap_ext = namedtuple('mmap', 'addr perms path rss anon locked')

    @wrap_exceptions
    def memory_maps(self):
        if False:
            for i in range(10):
                print('nop')

        def toaddr(start, end):
            if False:
                print('Hello World!')
            return '%s-%s' % (hex(start)[2:].strip('L'), hex(end)[2:].strip('L'))
        procfs_path = self._procfs_path
        retlist = []
        try:
            rawlist = cext.proc_memory_maps(self.pid, procfs_path)
        except OSError as err:
            if err.errno == errno.EOVERFLOW and (not IS_64_BIT):
                return []
            else:
                raise
        hit_enoent = False
        for item in rawlist:
            (addr, addrsize, perm, name, rss, anon, locked) = item
            addr = toaddr(addr, addrsize)
            if not name.startswith('['):
                try:
                    name = os.readlink('%s/%s/path/%s' % (procfs_path, self.pid, name))
                except OSError as err:
                    if err.errno == errno.ENOENT:
                        name = '%s/%s/path/%s' % (procfs_path, self.pid, name)
                        hit_enoent = True
                    else:
                        raise
            retlist.append((addr, perm, name, rss, anon, locked))
        if hit_enoent:
            self._assert_alive()
        return retlist

    @wrap_exceptions
    def num_fds(self):
        if False:
            while True:
                i = 10
        return len(os.listdir('%s/%s/fd' % (self._procfs_path, self.pid)))

    @wrap_exceptions
    def num_ctx_switches(self):
        if False:
            print('Hello World!')
        return _common.pctxsw(*cext.proc_num_ctx_switches(self.pid, self._procfs_path))

    @wrap_exceptions
    def wait(self, timeout=None):
        if False:
            return 10
        return _psposix.wait_pid(self.pid, timeout, self._name)