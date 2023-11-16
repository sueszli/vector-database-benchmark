"""Linux platform implementation."""
from __future__ import division
import base64
import collections
import errno
import functools
import glob
import os
import re
import socket
import struct
import sys
import warnings
from collections import defaultdict
from collections import namedtuple
from . import _common
from . import _psposix
from . import _psutil_linux as cext
from . import _psutil_posix as cext_posix
from ._common import NIC_DUPLEX_FULL
from ._common import NIC_DUPLEX_HALF
from ._common import NIC_DUPLEX_UNKNOWN
from ._common import AccessDenied
from ._common import NoSuchProcess
from ._common import ZombieProcess
from ._common import bcat
from ._common import cat
from ._common import debug
from ._common import decode
from ._common import get_procfs_path
from ._common import isfile_strict
from ._common import memoize
from ._common import memoize_when_activated
from ._common import open_binary
from ._common import open_text
from ._common import parse_environ_block
from ._common import path_exists_strict
from ._common import supports_ipv6
from ._common import usage_percent
from ._compat import PY3
from ._compat import FileNotFoundError
from ._compat import PermissionError
from ._compat import ProcessLookupError
from ._compat import b
from ._compat import basestring
if PY3:
    import enum
else:
    enum = None
__extra__all__ = ['PROCFS_PATH', 'IOPRIO_CLASS_NONE', 'IOPRIO_CLASS_RT', 'IOPRIO_CLASS_BE', 'IOPRIO_CLASS_IDLE', 'CONN_ESTABLISHED', 'CONN_SYN_SENT', 'CONN_SYN_RECV', 'CONN_FIN_WAIT1', 'CONN_FIN_WAIT2', 'CONN_TIME_WAIT', 'CONN_CLOSE', 'CONN_CLOSE_WAIT', 'CONN_LAST_ACK', 'CONN_LISTEN', 'CONN_CLOSING']
POWER_SUPPLY_PATH = '/sys/class/power_supply'
HAS_PROC_SMAPS = os.path.exists('/proc/%s/smaps' % os.getpid())
HAS_PROC_SMAPS_ROLLUP = os.path.exists('/proc/%s/smaps_rollup' % os.getpid())
HAS_PROC_IO_PRIORITY = hasattr(cext, 'proc_ioprio_get')
HAS_CPU_AFFINITY = hasattr(cext, 'proc_cpu_affinity_get')
CLOCK_TICKS = os.sysconf('SC_CLK_TCK')
PAGESIZE = cext_posix.getpagesize()
BOOT_TIME = None
LITTLE_ENDIAN = sys.byteorder == 'little'
DISK_SECTOR_SIZE = 512
if enum is None:
    AF_LINK = socket.AF_PACKET
else:
    AddressFamily = enum.IntEnum('AddressFamily', {'AF_LINK': int(socket.AF_PACKET)})
    AF_LINK = AddressFamily.AF_LINK
if enum is None:
    IOPRIO_CLASS_NONE = 0
    IOPRIO_CLASS_RT = 1
    IOPRIO_CLASS_BE = 2
    IOPRIO_CLASS_IDLE = 3
else:

    class IOPriority(enum.IntEnum):
        IOPRIO_CLASS_NONE = 0
        IOPRIO_CLASS_RT = 1
        IOPRIO_CLASS_BE = 2
        IOPRIO_CLASS_IDLE = 3
    globals().update(IOPriority.__members__)
PROC_STATUSES = {'R': _common.STATUS_RUNNING, 'S': _common.STATUS_SLEEPING, 'D': _common.STATUS_DISK_SLEEP, 'T': _common.STATUS_STOPPED, 't': _common.STATUS_TRACING_STOP, 'Z': _common.STATUS_ZOMBIE, 'X': _common.STATUS_DEAD, 'x': _common.STATUS_DEAD, 'K': _common.STATUS_WAKE_KILL, 'W': _common.STATUS_WAKING, 'I': _common.STATUS_IDLE, 'P': _common.STATUS_PARKED}
TCP_STATUSES = {'01': _common.CONN_ESTABLISHED, '02': _common.CONN_SYN_SENT, '03': _common.CONN_SYN_RECV, '04': _common.CONN_FIN_WAIT1, '05': _common.CONN_FIN_WAIT2, '06': _common.CONN_TIME_WAIT, '07': _common.CONN_CLOSE, '08': _common.CONN_CLOSE_WAIT, '09': _common.CONN_LAST_ACK, '0A': _common.CONN_LISTEN, '0B': _common.CONN_CLOSING}
svmem = namedtuple('svmem', ['total', 'available', 'percent', 'used', 'free', 'active', 'inactive', 'buffers', 'cached', 'shared', 'slab'])
sdiskio = namedtuple('sdiskio', ['read_count', 'write_count', 'read_bytes', 'write_bytes', 'read_time', 'write_time', 'read_merged_count', 'write_merged_count', 'busy_time'])
popenfile = namedtuple('popenfile', ['path', 'fd', 'position', 'mode', 'flags'])
pmem = namedtuple('pmem', 'rss vms shared text lib data dirty')
pfullmem = namedtuple('pfullmem', pmem._fields + ('uss', 'pss', 'swap'))
pmmap_grouped = namedtuple('pmmap_grouped', ['path', 'rss', 'size', 'pss', 'shared_clean', 'shared_dirty', 'private_clean', 'private_dirty', 'referenced', 'anonymous', 'swap'])
pmmap_ext = namedtuple('pmmap_ext', 'addr perms ' + ' '.join(pmmap_grouped._fields))
pio = namedtuple('pio', ['read_count', 'write_count', 'read_bytes', 'write_bytes', 'read_chars', 'write_chars'])
pcputimes = namedtuple('pcputimes', ['user', 'system', 'children_user', 'children_system', 'iowait'])

def readlink(path):
    if False:
        print('Hello World!')
    'Wrapper around os.readlink().'
    assert isinstance(path, basestring), path
    path = os.readlink(path)
    path = path.split('\x00')[0]
    if path.endswith(' (deleted)') and (not path_exists_strict(path)):
        path = path[:-10]
    return path

def file_flags_to_mode(flags):
    if False:
        for i in range(10):
            print('nop')
    "Convert file's open() flags into a readable string.\n    Used by Process.open_files().\n    "
    modes_map = {os.O_RDONLY: 'r', os.O_WRONLY: 'w', os.O_RDWR: 'w+'}
    mode = modes_map[flags & (os.O_RDONLY | os.O_WRONLY | os.O_RDWR)]
    if flags & os.O_APPEND:
        mode = mode.replace('w', 'a', 1)
    mode = mode.replace('w+', 'r+')
    return mode

def is_storage_device(name):
    if False:
        i = 10
        return i + 15
    'Return True if the given name refers to a root device (e.g.\n    "sda", "nvme0n1") as opposed to a logical partition (e.g.  "sda1",\n    "nvme0n1p1"). If name is a virtual device (e.g. "loop1", "ram")\n    return True.\n    '
    name = name.replace('/', '!')
    including_virtual = True
    if including_virtual:
        path = '/sys/block/%s' % name
    else:
        path = '/sys/block/%s/device' % name
    return os.access(path, os.F_OK)

@memoize
def set_scputimes_ntuple(procfs_path):
    if False:
        for i in range(10):
            print('nop')
    'Set a namedtuple of variable fields depending on the CPU times\n    available on this Linux kernel version which may be:\n    (user, nice, system, idle, iowait, irq, softirq, [steal, [guest,\n     [guest_nice]]])\n    Used by cpu_times() function.\n    '
    global scputimes
    with open_binary('%s/stat' % procfs_path) as f:
        values = f.readline().split()[1:]
    fields = ['user', 'nice', 'system', 'idle', 'iowait', 'irq', 'softirq']
    vlen = len(values)
    if vlen >= 8:
        fields.append('steal')
    if vlen >= 9:
        fields.append('guest')
    if vlen >= 10:
        fields.append('guest_nice')
    scputimes = namedtuple('scputimes', fields)
try:
    set_scputimes_ntuple('/proc')
except Exception as err:
    debug('ignoring exception on import: %r' % err)
    scputimes = namedtuple('scputimes', 'user system idle')(0.0, 0.0, 0.0)
prlimit = None
try:
    from resource import prlimit
except ImportError:
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    if hasattr(libc, 'prlimit'):

        def prlimit(pid, resource_, limits=None):
            if False:
                while True:
                    i = 10

            class StructRlimit(ctypes.Structure):
                _fields_ = [('rlim_cur', ctypes.c_longlong), ('rlim_max', ctypes.c_longlong)]
            current = StructRlimit()
            if limits is None:
                ret = libc.prlimit(pid, resource_, None, ctypes.byref(current))
            else:
                new = StructRlimit()
                new.rlim_cur = limits[0]
                new.rlim_max = limits[1]
                ret = libc.prlimit(pid, resource_, ctypes.byref(new), ctypes.byref(current))
            if ret != 0:
                errno_ = ctypes.get_errno()
                raise OSError(errno_, os.strerror(errno_))
            return (current.rlim_cur, current.rlim_max)
if prlimit is not None:
    __extra__all__.extend([x for x in dir(cext) if x.startswith('RLIM') and x.isupper()])

def calculate_avail_vmem(mems):
    if False:
        return 10
    'Fallback for kernels < 3.14 where /proc/meminfo does not provide\n    "MemAvailable", see:\n    https://blog.famzah.net/2014/09/24/.\n\n    This code reimplements the algorithm outlined here:\n    https://git.kernel.org/cgit/linux/kernel/git/torvalds/linux.git/\n        commit/?id=34e431b0ae398fc54ea69ff85ec700722c9da773\n\n    We use this function also when "MemAvailable" returns 0 (possibly a\n    kernel bug, see: https://github.com/giampaolo/psutil/issues/1915).\n    In that case this routine matches "free" CLI tool result ("available"\n    column).\n\n    XXX: on recent kernels this calculation may differ by ~1.5% compared\n    to "MemAvailable:", as it\'s calculated slightly differently.\n    It is still way more realistic than doing (free + cached) though.\n    See:\n    * https://gitlab.com/procps-ng/procps/issues/42\n    * https://github.com/famzah/linux-memavailable-procfs/issues/2\n    '
    free = mems[b'MemFree:']
    fallback = free + mems.get(b'Cached:', 0)
    try:
        lru_active_file = mems[b'Active(file):']
        lru_inactive_file = mems[b'Inactive(file):']
        slab_reclaimable = mems[b'SReclaimable:']
    except KeyError as err:
        debug('%r is missing from /proc/meminfo; using an approximation for calculating available memory' % err.args[0])
        return fallback
    try:
        f = open_binary('%s/zoneinfo' % get_procfs_path())
    except IOError:
        return fallback
    watermark_low = 0
    with f:
        for line in f:
            line = line.strip()
            if line.startswith(b'low'):
                watermark_low += int(line.split()[1])
    watermark_low *= PAGESIZE
    avail = free - watermark_low
    pagecache = lru_active_file + lru_inactive_file
    pagecache -= min(pagecache / 2, watermark_low)
    avail += pagecache
    avail += slab_reclaimable - min(slab_reclaimable / 2.0, watermark_low)
    return int(avail)

def virtual_memory():
    if False:
        for i in range(10):
            print('nop')
    'Report virtual memory stats.\n    This implementation mimicks procps-ng-3.3.12, aka "free" CLI tool:\n    https://gitlab.com/procps-ng/procps/blob/\n        24fd2605c51fccc375ab0287cec33aa767f06718/proc/sysinfo.c#L778-791\n    The returned values are supposed to match both "free" and "vmstat -s"\n    CLI tools.\n    '
    missing_fields = []
    mems = {}
    with open_binary('%s/meminfo' % get_procfs_path()) as f:
        for line in f:
            fields = line.split()
            mems[fields[0]] = int(fields[1]) * 1024
    total = mems[b'MemTotal:']
    free = mems[b'MemFree:']
    try:
        buffers = mems[b'Buffers:']
    except KeyError:
        buffers = 0
        missing_fields.append('buffers')
    try:
        cached = mems[b'Cached:']
    except KeyError:
        cached = 0
        missing_fields.append('cached')
    else:
        cached += mems.get(b'SReclaimable:', 0)
    try:
        shared = mems[b'Shmem:']
    except KeyError:
        try:
            shared = mems[b'MemShared:']
        except KeyError:
            shared = 0
            missing_fields.append('shared')
    try:
        active = mems[b'Active:']
    except KeyError:
        active = 0
        missing_fields.append('active')
    try:
        inactive = mems[b'Inactive:']
    except KeyError:
        try:
            inactive = mems[b'Inact_dirty:'] + mems[b'Inact_clean:'] + mems[b'Inact_laundry:']
        except KeyError:
            inactive = 0
            missing_fields.append('inactive')
    try:
        slab = mems[b'Slab:']
    except KeyError:
        slab = 0
    used = total - free - cached - buffers
    if used < 0:
        used = total - free
    try:
        avail = mems[b'MemAvailable:']
    except KeyError:
        avail = calculate_avail_vmem(mems)
    else:
        if avail == 0:
            avail = calculate_avail_vmem(mems)
    if avail < 0:
        avail = 0
        missing_fields.append('available')
    elif avail > total:
        avail = free
    percent = usage_percent(total - avail, total, round_=1)
    if missing_fields:
        msg = "%s memory stats couldn't be determined and %s set to 0" % (', '.join(missing_fields), 'was' if len(missing_fields) == 1 else 'were')
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    return svmem(total, avail, percent, used, free, active, inactive, buffers, cached, shared, slab)

def swap_memory():
    if False:
        return 10
    'Return swap memory metrics.'
    mems = {}
    with open_binary('%s/meminfo' % get_procfs_path()) as f:
        for line in f:
            fields = line.split()
            mems[fields[0]] = int(fields[1]) * 1024
    try:
        total = mems[b'SwapTotal:']
        free = mems[b'SwapFree:']
    except KeyError:
        (_, _, _, _, total, free, unit_multiplier) = cext.linux_sysinfo()
        total *= unit_multiplier
        free *= unit_multiplier
    used = total - free
    percent = usage_percent(used, total, round_=1)
    try:
        f = open_binary('%s/vmstat' % get_procfs_path())
    except IOError as err:
        msg = "'sin' and 'sout' swap memory stats couldn't " + 'be determined and were set to 0 (%s)' % str(err)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        sin = sout = 0
    else:
        with f:
            sin = sout = None
            for line in f:
                if line.startswith(b'pswpin'):
                    sin = int(line.split(b' ')[1]) * 4 * 1024
                elif line.startswith(b'pswpout'):
                    sout = int(line.split(b' ')[1]) * 4 * 1024
                if sin is not None and sout is not None:
                    break
            else:
                msg = "'sin' and 'sout' swap memory stats couldn't "
                msg += 'be determined and were set to 0'
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
                sin = sout = 0
    return _common.sswap(total, used, free, percent, sin, sout)

def cpu_times():
    if False:
        for i in range(10):
            print('nop')
    'Return a named tuple representing the following system-wide\n    CPU times:\n    (user, nice, system, idle, iowait, irq, softirq [steal, [guest,\n     [guest_nice]]])\n    Last 3 fields may not be available on all Linux kernel versions.\n    '
    procfs_path = get_procfs_path()
    set_scputimes_ntuple(procfs_path)
    with open_binary('%s/stat' % procfs_path) as f:
        values = f.readline().split()
    fields = values[1:len(scputimes._fields) + 1]
    fields = [float(x) / CLOCK_TICKS for x in fields]
    return scputimes(*fields)

def per_cpu_times():
    if False:
        i = 10
        return i + 15
    'Return a list of namedtuple representing the CPU times\n    for every CPU available on the system.\n    '
    procfs_path = get_procfs_path()
    set_scputimes_ntuple(procfs_path)
    cpus = []
    with open_binary('%s/stat' % procfs_path) as f:
        f.readline()
        for line in f:
            if line.startswith(b'cpu'):
                values = line.split()
                fields = values[1:len(scputimes._fields) + 1]
                fields = [float(x) / CLOCK_TICKS for x in fields]
                entry = scputimes(*fields)
                cpus.append(entry)
        return cpus

def cpu_count_logical():
    if False:
        for i in range(10):
            print('nop')
    'Return the number of logical CPUs in the system.'
    try:
        return os.sysconf('SC_NPROCESSORS_ONLN')
    except ValueError:
        num = 0
        with open_binary('%s/cpuinfo' % get_procfs_path()) as f:
            for line in f:
                if line.lower().startswith(b'processor'):
                    num += 1
        if num == 0:
            search = re.compile('cpu\\d')
            with open_text('%s/stat' % get_procfs_path()) as f:
                for line in f:
                    line = line.split(' ')[0]
                    if search.match(line):
                        num += 1
        if num == 0:
            return None
        return num

def cpu_count_cores():
    if False:
        print('Hello World!')
    'Return the number of CPU cores in the system.'
    ls = set()
    p1 = '/sys/devices/system/cpu/cpu[0-9]*/topology/core_cpus_list'
    p2 = '/sys/devices/system/cpu/cpu[0-9]*/topology/thread_siblings_list'
    for path in glob.glob(p1) or glob.glob(p2):
        with open_binary(path) as f:
            ls.add(f.read().strip())
    result = len(ls)
    if result != 0:
        return result
    mapping = {}
    current_info = {}
    with open_binary('%s/cpuinfo' % get_procfs_path()) as f:
        for line in f:
            line = line.strip().lower()
            if not line:
                try:
                    mapping[current_info[b'physical id']] = current_info[b'cpu cores']
                except KeyError:
                    pass
                current_info = {}
            elif line.startswith((b'physical id', b'cpu cores')):
                (key, value) = line.split(b'\t:', 1)
                current_info[key] = int(value)
    result = sum(mapping.values())
    return result or None

def cpu_stats():
    if False:
        for i in range(10):
            print('nop')
    'Return various CPU stats as a named tuple.'
    with open_binary('%s/stat' % get_procfs_path()) as f:
        ctx_switches = None
        interrupts = None
        soft_interrupts = None
        for line in f:
            if line.startswith(b'ctxt'):
                ctx_switches = int(line.split()[1])
            elif line.startswith(b'intr'):
                interrupts = int(line.split()[1])
            elif line.startswith(b'softirq'):
                soft_interrupts = int(line.split()[1])
            if ctx_switches is not None and soft_interrupts is not None and (interrupts is not None):
                break
    syscalls = 0
    return _common.scpustats(ctx_switches, interrupts, soft_interrupts, syscalls)

def _cpu_get_cpuinfo_freq():
    if False:
        while True:
            i = 10
    'Return current CPU frequency from cpuinfo if available.'
    ret = []
    with open_binary('%s/cpuinfo' % get_procfs_path()) as f:
        for line in f:
            if line.lower().startswith(b'cpu mhz'):
                ret.append(float(line.split(b':', 1)[1]))
    return ret
if os.path.exists('/sys/devices/system/cpu/cpufreq/policy0') or os.path.exists('/sys/devices/system/cpu/cpu0/cpufreq'):

    def cpu_freq():
        if False:
            return 10
        'Return frequency metrics for all CPUs.\n        Contrarily to other OSes, Linux updates these values in\n        real-time.\n        '
        cpuinfo_freqs = _cpu_get_cpuinfo_freq()
        paths = glob.glob('/sys/devices/system/cpu/cpufreq/policy[0-9]*') or glob.glob('/sys/devices/system/cpu/cpu[0-9]*/cpufreq')
        paths.sort(key=lambda x: int(re.search('[0-9]+', x).group()))
        ret = []
        pjoin = os.path.join
        for (i, path) in enumerate(paths):
            if len(paths) == len(cpuinfo_freqs):
                curr = cpuinfo_freqs[i] * 1000
            else:
                curr = bcat(pjoin(path, 'scaling_cur_freq'), fallback=None)
            if curr is None:
                curr = bcat(pjoin(path, 'cpuinfo_cur_freq'), fallback=None)
                if curr is None:
                    msg = "can't find current frequency file"
                    raise NotImplementedError(msg)
            curr = int(curr) / 1000
            max_ = int(bcat(pjoin(path, 'scaling_max_freq'))) / 1000
            min_ = int(bcat(pjoin(path, 'scaling_min_freq'))) / 1000
            ret.append(_common.scpufreq(curr, min_, max_))
        return ret
else:

    def cpu_freq():
        if False:
            i = 10
            return i + 15
        'Alternate implementation using /proc/cpuinfo.\n        min and max frequencies are not available and are set to None.\n        '
        return [_common.scpufreq(x, 0.0, 0.0) for x in _cpu_get_cpuinfo_freq()]
net_if_addrs = cext_posix.net_if_addrs

class _Ipv6UnsupportedError(Exception):
    pass

class Connections:
    """A wrapper on top of /proc/net/* files, retrieving per-process
    and system-wide open connections (TCP, UDP, UNIX) similarly to
    "netstat -an".

    Note: in case of UNIX sockets we're only able to determine the
    local endpoint/path, not the one it's connected to.
    According to [1] it would be possible but not easily.

    [1] http://serverfault.com/a/417946
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        tcp4 = ('tcp', socket.AF_INET, socket.SOCK_STREAM)
        tcp6 = ('tcp6', socket.AF_INET6, socket.SOCK_STREAM)
        udp4 = ('udp', socket.AF_INET, socket.SOCK_DGRAM)
        udp6 = ('udp6', socket.AF_INET6, socket.SOCK_DGRAM)
        unix = ('unix', socket.AF_UNIX, None)
        self.tmap = {'all': (tcp4, tcp6, udp4, udp6, unix), 'tcp': (tcp4, tcp6), 'tcp4': (tcp4,), 'tcp6': (tcp6,), 'udp': (udp4, udp6), 'udp4': (udp4,), 'udp6': (udp6,), 'unix': (unix,), 'inet': (tcp4, tcp6, udp4, udp6), 'inet4': (tcp4, udp4), 'inet6': (tcp6, udp6)}
        self._procfs_path = None

    def get_proc_inodes(self, pid):
        if False:
            while True:
                i = 10
        inodes = defaultdict(list)
        for fd in os.listdir('%s/%s/fd' % (self._procfs_path, pid)):
            try:
                inode = readlink('%s/%s/fd/%s' % (self._procfs_path, pid, fd))
            except (FileNotFoundError, ProcessLookupError):
                continue
            except OSError as err:
                if err.errno == errno.EINVAL:
                    continue
                if err.errno == errno.ENAMETOOLONG:
                    debug(err)
                    continue
                raise
            else:
                if inode.startswith('socket:['):
                    inode = inode[8:][:-1]
                    inodes[inode].append((pid, int(fd)))
        return inodes

    def get_all_inodes(self):
        if False:
            return 10
        inodes = {}
        for pid in pids():
            try:
                inodes.update(self.get_proc_inodes(pid))
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
        return inodes

    @staticmethod
    def decode_address(addr, family):
        if False:
            for i in range(10):
                print('nop')
        'Accept an "ip:port" address as displayed in /proc/net/*\n        and convert it into a human readable form, like:\n\n        "0500000A:0016" -> ("10.0.0.5", 22)\n        "0000000000000000FFFF00000100007F:9E49" -> ("::ffff:127.0.0.1", 40521)\n\n        The IP address portion is a little or big endian four-byte\n        hexadecimal number; that is, the least significant byte is listed\n        first, so we need to reverse the order of the bytes to convert it\n        to an IP address.\n        The port is represented as a two-byte hexadecimal number.\n\n        Reference:\n        http://linuxdevcenter.com/pub/a/linux/2000/11/16/LinuxAdmin.html\n        '
        (ip, port) = addr.split(':')
        port = int(port, 16)
        if not port:
            return ()
        if PY3:
            ip = ip.encode('ascii')
        if family == socket.AF_INET:
            if LITTLE_ENDIAN:
                ip = socket.inet_ntop(family, base64.b16decode(ip)[::-1])
            else:
                ip = socket.inet_ntop(family, base64.b16decode(ip))
        else:
            ip = base64.b16decode(ip)
            try:
                if LITTLE_ENDIAN:
                    ip = socket.inet_ntop(socket.AF_INET6, struct.pack('>4I', *struct.unpack('<4I', ip)))
                else:
                    ip = socket.inet_ntop(socket.AF_INET6, struct.pack('<4I', *struct.unpack('<4I', ip)))
            except ValueError:
                if not supports_ipv6():
                    raise _Ipv6UnsupportedError
                else:
                    raise
        return _common.addr(ip, port)

    @staticmethod
    def process_inet(file, family, type_, inodes, filter_pid=None):
        if False:
            i = 10
            return i + 15
        'Parse /proc/net/tcp* and /proc/net/udp* files.'
        if file.endswith('6') and (not os.path.exists(file)):
            return
        with open_text(file) as f:
            f.readline()
            for (lineno, line) in enumerate(f, 1):
                try:
                    (_, laddr, raddr, status, _, _, _, _, _, inode) = line.split()[:10]
                except ValueError:
                    raise RuntimeError('error while parsing %s; malformed line %s %r' % (file, lineno, line))
                if inode in inodes:
                    (pid, fd) = inodes[inode][0]
                else:
                    (pid, fd) = (None, -1)
                if filter_pid is not None and filter_pid != pid:
                    continue
                else:
                    if type_ == socket.SOCK_STREAM:
                        status = TCP_STATUSES[status]
                    else:
                        status = _common.CONN_NONE
                    try:
                        laddr = Connections.decode_address(laddr, family)
                        raddr = Connections.decode_address(raddr, family)
                    except _Ipv6UnsupportedError:
                        continue
                    yield (fd, family, type_, laddr, raddr, status, pid)

    @staticmethod
    def process_unix(file, family, inodes, filter_pid=None):
        if False:
            i = 10
            return i + 15
        'Parse /proc/net/unix files.'
        with open_text(file) as f:
            f.readline()
            for line in f:
                tokens = line.split()
                try:
                    (_, _, _, _, type_, _, inode) = tokens[0:7]
                except ValueError:
                    if ' ' not in line:
                        continue
                    raise RuntimeError('error while parsing %s; malformed line %r' % (file, line))
                if inode in inodes:
                    pairs = inodes[inode]
                else:
                    pairs = [(None, -1)]
                for (pid, fd) in pairs:
                    if filter_pid is not None and filter_pid != pid:
                        continue
                    else:
                        path = tokens[-1] if len(tokens) == 8 else ''
                        type_ = _common.socktype_to_enum(int(type_))
                        raddr = ''
                        status = _common.CONN_NONE
                        yield (fd, family, type_, path, raddr, status, pid)

    def retrieve(self, kind, pid=None):
        if False:
            while True:
                i = 10
        if kind not in self.tmap:
            raise ValueError('invalid %r kind argument; choose between %s' % (kind, ', '.join([repr(x) for x in self.tmap])))
        self._procfs_path = get_procfs_path()
        if pid is not None:
            inodes = self.get_proc_inodes(pid)
            if not inodes:
                return []
        else:
            inodes = self.get_all_inodes()
        ret = set()
        for (proto_name, family, type_) in self.tmap[kind]:
            path = '%s/net/%s' % (self._procfs_path, proto_name)
            if family in (socket.AF_INET, socket.AF_INET6):
                ls = self.process_inet(path, family, type_, inodes, filter_pid=pid)
            else:
                ls = self.process_unix(path, family, inodes, filter_pid=pid)
            for (fd, family, type_, laddr, raddr, status, bound_pid) in ls:
                if pid:
                    conn = _common.pconn(fd, family, type_, laddr, raddr, status)
                else:
                    conn = _common.sconn(fd, family, type_, laddr, raddr, status, bound_pid)
                ret.add(conn)
        return list(ret)
_connections = Connections()

def net_connections(kind='inet'):
    if False:
        i = 10
        return i + 15
    'Return system-wide open connections.'
    return _connections.retrieve(kind)

def net_io_counters():
    if False:
        return 10
    'Return network I/O statistics for every network interface\n    installed on the system as a dict of raw tuples.\n    '
    with open_text('%s/net/dev' % get_procfs_path()) as f:
        lines = f.readlines()
    retdict = {}
    for line in lines[2:]:
        colon = line.rfind(':')
        assert colon > 0, repr(line)
        name = line[:colon].strip()
        fields = line[colon + 1:].strip().split()
        (bytes_recv, packets_recv, errin, dropin, fifoin, framein, compressedin, multicastin, bytes_sent, packets_sent, errout, dropout, fifoout, collisionsout, carrierout, compressedout) = map(int, fields)
        retdict[name] = (bytes_sent, bytes_recv, packets_sent, packets_recv, errin, errout, dropin, dropout)
    return retdict

def net_if_stats():
    if False:
        return 10
    'Get NIC stats (isup, duplex, speed, mtu).'
    duplex_map = {cext.DUPLEX_FULL: NIC_DUPLEX_FULL, cext.DUPLEX_HALF: NIC_DUPLEX_HALF, cext.DUPLEX_UNKNOWN: NIC_DUPLEX_UNKNOWN}
    names = net_io_counters().keys()
    ret = {}
    for name in names:
        try:
            mtu = cext_posix.net_if_mtu(name)
            flags = cext_posix.net_if_flags(name)
            (duplex, speed) = cext.net_if_duplex_speed(name)
        except OSError as err:
            if err.errno != errno.ENODEV:
                raise
            else:
                debug(err)
        else:
            output_flags = ','.join(flags)
            isup = 'running' in flags
            ret[name] = _common.snicstats(isup, duplex_map[duplex], speed, mtu, output_flags)
    return ret
disk_usage = _psposix.disk_usage

def disk_io_counters(perdisk=False):
    if False:
        print('Hello World!')
    'Return disk I/O statistics for every disk installed on the\n    system as a dict of raw tuples.\n    '

    def read_procfs():
        if False:
            for i in range(10):
                print('nop')
        with open_text('%s/diskstats' % get_procfs_path()) as f:
            lines = f.readlines()
        for line in lines:
            fields = line.split()
            flen = len(fields)
            if flen == 15:
                name = fields[3]
                reads = int(fields[2])
                (reads_merged, rbytes, rtime, writes, writes_merged, wbytes, wtime, _, busy_time, _) = map(int, fields[4:14])
            elif flen == 14 or flen >= 18:
                name = fields[2]
                (reads, reads_merged, rbytes, rtime, writes, writes_merged, wbytes, wtime, _, busy_time, _) = map(int, fields[3:14])
            elif flen == 7:
                name = fields[2]
                (reads, rbytes, writes, wbytes) = map(int, fields[3:])
                rtime = wtime = reads_merged = writes_merged = busy_time = 0
            else:
                raise ValueError('not sure how to interpret line %r' % line)
            yield (name, reads, writes, rbytes, wbytes, rtime, wtime, reads_merged, writes_merged, busy_time)

    def read_sysfs():
        if False:
            return 10
        for block in os.listdir('/sys/block'):
            for (root, _, files) in os.walk(os.path.join('/sys/block', block)):
                if 'stat' not in files:
                    continue
                with open_text(os.path.join(root, 'stat')) as f:
                    fields = f.read().strip().split()
                name = os.path.basename(root)
                (reads, reads_merged, rbytes, rtime, writes, writes_merged, wbytes, wtime, _, busy_time) = map(int, fields[:10])
                yield (name, reads, writes, rbytes, wbytes, rtime, wtime, reads_merged, writes_merged, busy_time)
    if os.path.exists('%s/diskstats' % get_procfs_path()):
        gen = read_procfs()
    elif os.path.exists('/sys/block'):
        gen = read_sysfs()
    else:
        raise NotImplementedError('%s/diskstats nor /sys/block filesystem are available on this system' % get_procfs_path())
    retdict = {}
    for entry in gen:
        (name, reads, writes, rbytes, wbytes, rtime, wtime, reads_merged, writes_merged, busy_time) = entry
        if not perdisk and (not is_storage_device(name)):
            continue
        rbytes *= DISK_SECTOR_SIZE
        wbytes *= DISK_SECTOR_SIZE
        retdict[name] = (reads, writes, rbytes, wbytes, rtime, wtime, reads_merged, writes_merged, busy_time)
    return retdict

class RootFsDeviceFinder:
    """disk_partitions() may return partitions with device == "/dev/root"
    or "rootfs". This container class uses different strategies to try to
    obtain the real device path. Resources:
    https://bootlin.com/blog/find-root-device/
    https://www.systutorials.com/how-to-find-the-disk-where-root-is-on-in-bash-on-linux/.
    """
    __slots__ = ['major', 'minor']

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        dev = os.stat('/').st_dev
        self.major = os.major(dev)
        self.minor = os.minor(dev)

    def ask_proc_partitions(self):
        if False:
            i = 10
            return i + 15
        with open_text('%s/partitions' % get_procfs_path()) as f:
            for line in f.readlines()[2:]:
                fields = line.split()
                if len(fields) < 4:
                    continue
                major = int(fields[0]) if fields[0].isdigit() else None
                minor = int(fields[1]) if fields[1].isdigit() else None
                name = fields[3]
                if major == self.major and minor == self.minor:
                    if name:
                        return '/dev/%s' % name

    def ask_sys_dev_block(self):
        if False:
            return 10
        path = '/sys/dev/block/%s:%s/uevent' % (self.major, self.minor)
        with open_text(path) as f:
            for line in f:
                if line.startswith('DEVNAME='):
                    name = line.strip().rpartition('DEVNAME=')[2]
                    if name:
                        return '/dev/%s' % name

    def ask_sys_class_block(self):
        if False:
            for i in range(10):
                print('nop')
        needle = '%s:%s' % (self.major, self.minor)
        files = glob.iglob('/sys/class/block/*/dev')
        for file in files:
            try:
                f = open_text(file)
            except FileNotFoundError:
                continue
            else:
                with f:
                    data = f.read().strip()
                    if data == needle:
                        name = os.path.basename(os.path.dirname(file))
                        return '/dev/%s' % name

    def find(self):
        if False:
            for i in range(10):
                print('nop')
        path = None
        if path is None:
            try:
                path = self.ask_proc_partitions()
            except (IOError, OSError) as err:
                debug(err)
        if path is None:
            try:
                path = self.ask_sys_dev_block()
            except (IOError, OSError) as err:
                debug(err)
        if path is None:
            try:
                path = self.ask_sys_class_block()
            except (IOError, OSError) as err:
                debug(err)
        if path is not None and os.path.exists(path):
            return path

def disk_partitions(all=False):
    if False:
        i = 10
        return i + 15
    'Return mounted disk partitions as a list of namedtuples.'
    fstypes = set()
    procfs_path = get_procfs_path()
    if not all:
        with open_text('%s/filesystems' % procfs_path) as f:
            for line in f:
                line = line.strip()
                if not line.startswith('nodev'):
                    fstypes.add(line.strip())
                else:
                    fstype = line.split('\t')[1]
                    if fstype == 'zfs':
                        fstypes.add('zfs')
    if procfs_path == '/proc' and os.path.isfile('/etc/mtab'):
        mounts_path = os.path.realpath('/etc/mtab')
    else:
        mounts_path = os.path.realpath('%s/self/mounts' % procfs_path)
    retlist = []
    partitions = cext.disk_partitions(mounts_path)
    for partition in partitions:
        (device, mountpoint, fstype, opts) = partition
        if device == 'none':
            device = ''
        if device in ('/dev/root', 'rootfs'):
            device = RootFsDeviceFinder().find() or device
        if not all:
            if device == '' or fstype not in fstypes:
                continue
        maxfile = maxpath = None
        ntuple = _common.sdiskpart(device, mountpoint, fstype, opts, maxfile, maxpath)
        retlist.append(ntuple)
    return retlist

def sensors_temperatures():
    if False:
        return 10
    "Return hardware (CPU and others) temperatures as a dict\n    including hardware name, label, current, max and critical\n    temperatures.\n\n    Implementation notes:\n    - /sys/class/hwmon looks like the most recent interface to\n      retrieve this info, and this implementation relies on it\n      only (old distros will probably use something else)\n    - lm-sensors on Ubuntu 16.04 relies on /sys/class/hwmon\n    - /sys/class/thermal/thermal_zone* is another one but it's more\n      difficult to parse\n    "
    ret = collections.defaultdict(list)
    basenames = glob.glob('/sys/class/hwmon/hwmon*/temp*_*')
    basenames.extend(glob.glob('/sys/class/hwmon/hwmon*/device/temp*_*'))
    basenames = sorted(set([x.split('_')[0] for x in basenames]))
    basenames2 = glob.glob('/sys/devices/platform/coretemp.*/hwmon/hwmon*/temp*_*')
    repl = re.compile('/sys/devices/platform/coretemp.*/hwmon/')
    for name in basenames2:
        altname = repl.sub('/sys/class/hwmon/', name)
        if altname not in basenames:
            basenames.append(name)
    for base in basenames:
        try:
            path = base + '_input'
            current = float(bcat(path)) / 1000.0
            path = os.path.join(os.path.dirname(base), 'name')
            unit_name = cat(path).strip()
        except (IOError, OSError, ValueError):
            continue
        high = bcat(base + '_max', fallback=None)
        critical = bcat(base + '_crit', fallback=None)
        label = cat(base + '_label', fallback='').strip()
        if high is not None:
            try:
                high = float(high) / 1000.0
            except ValueError:
                high = None
        if critical is not None:
            try:
                critical = float(critical) / 1000.0
            except ValueError:
                critical = None
        ret[unit_name].append((label, current, high, critical))
    if not basenames:
        basenames = glob.glob('/sys/class/thermal/thermal_zone*')
        basenames = sorted(set(basenames))
        for base in basenames:
            try:
                path = os.path.join(base, 'temp')
                current = float(bcat(path)) / 1000.0
                path = os.path.join(base, 'type')
                unit_name = cat(path).strip()
            except (IOError, OSError, ValueError) as err:
                debug(err)
                continue
            trip_paths = glob.glob(base + '/trip_point*')
            trip_points = set(['_'.join(os.path.basename(p).split('_')[0:3]) for p in trip_paths])
            critical = None
            high = None
            for trip_point in trip_points:
                path = os.path.join(base, trip_point + '_type')
                trip_type = cat(path, fallback='').strip()
                if trip_type == 'critical':
                    critical = bcat(os.path.join(base, trip_point + '_temp'), fallback=None)
                elif trip_type == 'high':
                    high = bcat(os.path.join(base, trip_point + '_temp'), fallback=None)
                if high is not None:
                    try:
                        high = float(high) / 1000.0
                    except ValueError:
                        high = None
                if critical is not None:
                    try:
                        critical = float(critical) / 1000.0
                    except ValueError:
                        critical = None
            ret[unit_name].append(('', current, high, critical))
    return dict(ret)

def sensors_fans():
    if False:
        i = 10
        return i + 15
    'Return hardware fans info (for CPU and other peripherals) as a\n    dict including hardware label and current speed.\n\n    Implementation notes:\n    - /sys/class/hwmon looks like the most recent interface to\n      retrieve this info, and this implementation relies on it\n      only (old distros will probably use something else)\n    - lm-sensors on Ubuntu 16.04 relies on /sys/class/hwmon\n    '
    ret = collections.defaultdict(list)
    basenames = glob.glob('/sys/class/hwmon/hwmon*/fan*_*')
    if not basenames:
        basenames = glob.glob('/sys/class/hwmon/hwmon*/device/fan*_*')
    basenames = sorted(set([x.split('_')[0] for x in basenames]))
    for base in basenames:
        try:
            current = int(bcat(base + '_input'))
        except (IOError, OSError) as err:
            debug(err)
            continue
        unit_name = cat(os.path.join(os.path.dirname(base), 'name')).strip()
        label = cat(base + '_label', fallback='').strip()
        ret[unit_name].append(_common.sfan(label, current))
    return dict(ret)

def sensors_battery():
    if False:
        return 10
    'Return battery information.\n    Implementation note: it appears /sys/class/power_supply/BAT0/\n    directory structure may vary and provide files with the same\n    meaning but under different names, see:\n    https://github.com/giampaolo/psutil/issues/966.\n    '
    null = object()

    def multi_bcat(*paths):
        if False:
            for i in range(10):
                print('nop')
        'Attempt to read the content of multiple files which may\n        not exist. If none of them exist return None.\n        '
        for path in paths:
            ret = bcat(path, fallback=null)
            if ret != null:
                try:
                    return int(ret)
                except ValueError:
                    return ret.strip()
        return None
    bats = [x for x in os.listdir(POWER_SUPPLY_PATH) if x.startswith('BAT') or 'battery' in x.lower()]
    if not bats:
        return None
    root = os.path.join(POWER_SUPPLY_PATH, sorted(bats)[0])
    energy_now = multi_bcat(root + '/energy_now', root + '/charge_now')
    power_now = multi_bcat(root + '/power_now', root + '/current_now')
    energy_full = multi_bcat(root + '/energy_full', root + '/charge_full')
    time_to_empty = multi_bcat(root + '/time_to_empty_now')
    if energy_full is not None and energy_now is not None:
        try:
            percent = 100.0 * energy_now / energy_full
        except ZeroDivisionError:
            percent = 0.0
    else:
        percent = int(cat(root + '/capacity', fallback=-1))
        if percent == -1:
            return None
    power_plugged = None
    online = multi_bcat(os.path.join(POWER_SUPPLY_PATH, 'AC0/online'), os.path.join(POWER_SUPPLY_PATH, 'AC/online'))
    if online is not None:
        power_plugged = online == 1
    else:
        status = cat(root + '/status', fallback='').strip().lower()
        if status == 'discharging':
            power_plugged = False
        elif status in ('charging', 'full'):
            power_plugged = True
    if power_plugged:
        secsleft = _common.POWER_TIME_UNLIMITED
    elif energy_now is not None and power_now is not None:
        try:
            secsleft = int(energy_now / power_now * 3600)
        except ZeroDivisionError:
            secsleft = _common.POWER_TIME_UNKNOWN
    elif time_to_empty is not None:
        secsleft = int(time_to_empty * 60)
        if secsleft < 0:
            secsleft = _common.POWER_TIME_UNKNOWN
    else:
        secsleft = _common.POWER_TIME_UNKNOWN
    return _common.sbattery(percent, secsleft, power_plugged)

def users():
    if False:
        return 10
    'Return currently connected users as a list of namedtuples.'
    retlist = []
    rawlist = cext.users()
    for item in rawlist:
        (user, tty, hostname, tstamp, pid) = item
        nt = _common.suser(user, tty or None, hostname, tstamp, pid)
        retlist.append(nt)
    return retlist

def boot_time():
    if False:
        i = 10
        return i + 15
    'Return the system boot time expressed in seconds since the epoch.'
    global BOOT_TIME
    path = '%s/stat' % get_procfs_path()
    with open_binary(path) as f:
        for line in f:
            if line.startswith(b'btime'):
                ret = float(line.strip().split()[1])
                BOOT_TIME = ret
                return ret
        raise RuntimeError("line 'btime' not found in %s" % path)

def pids():
    if False:
        while True:
            i = 10
    'Returns a list of PIDs currently running on the system.'
    return [int(x) for x in os.listdir(b(get_procfs_path())) if x.isdigit()]

def pid_exists(pid):
    if False:
        return 10
    'Check for the existence of a unix PID. Linux TIDs are not\n    supported (always return False).\n    '
    if not _psposix.pid_exists(pid):
        return False
    else:
        try:
            path = '%s/%s/status' % (get_procfs_path(), pid)
            with open_binary(path) as f:
                for line in f:
                    if line.startswith(b'Tgid:'):
                        tgid = int(line.split()[1])
                        return tgid == pid
                raise ValueError("'Tgid' line not found in %s" % path)
        except (EnvironmentError, ValueError):
            return pid in pids()

def ppid_map():
    if False:
        print('Hello World!')
    'Obtain a {pid: ppid, ...} dict for all running processes in\n    one shot. Used to speed up Process.children().\n    '
    ret = {}
    procfs_path = get_procfs_path()
    for pid in pids():
        try:
            with open_binary('%s/%s/stat' % (procfs_path, pid)) as f:
                data = f.read()
        except (FileNotFoundError, ProcessLookupError):
            pass
        else:
            rpar = data.rfind(b')')
            dset = data[rpar + 2:].split()
            ppid = int(dset[1])
            ret[pid] = ppid
    return ret

def wrap_exceptions(fun):
    if False:
        for i in range(10):
            print('nop')
    'Decorator which translates bare OSError and IOError exceptions\n    into NoSuchProcess and AccessDenied.\n    '

    @functools.wraps(fun)
    def wrapper(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        try:
            return fun(self, *args, **kwargs)
        except PermissionError:
            raise AccessDenied(self.pid, self._name)
        except ProcessLookupError:
            self._raise_if_zombie()
            raise NoSuchProcess(self.pid, self._name)
        except FileNotFoundError:
            self._raise_if_zombie()
            if not os.path.exists('%s/%s' % (self._procfs_path, self.pid)):
                raise NoSuchProcess(self.pid, self._name)
            raise
    return wrapper

class Process:
    """Linux process implementation."""
    __slots__ = ['pid', '_name', '_ppid', '_procfs_path', '_cache']

    def __init__(self, pid):
        if False:
            print('Hello World!')
        self.pid = pid
        self._name = None
        self._ppid = None
        self._procfs_path = get_procfs_path()

    def _is_zombie(self):
        if False:
            print('Hello World!')
        try:
            data = bcat('%s/%s/stat' % (self._procfs_path, self.pid))
        except (IOError, OSError):
            return False
        else:
            rpar = data.rfind(b')')
            status = data[rpar + 2:rpar + 3]
            return status == b'Z'

    def _raise_if_zombie(self):
        if False:
            for i in range(10):
                print('nop')
        if self._is_zombie():
            raise ZombieProcess(self.pid, self._name, self._ppid)

    def _raise_if_not_alive(self):
        if False:
            print('Hello World!')
        'Raise NSP if the process disappeared on us.'
        os.stat('%s/%s' % (self._procfs_path, self.pid))

    @wrap_exceptions
    @memoize_when_activated
    def _parse_stat_file(self):
        if False:
            print('Hello World!')
        'Parse /proc/{pid}/stat file and return a dict with various\n        process info.\n        Using "man proc" as a reference: where "man proc" refers to\n        position N always subtract 3 (e.g ppid position 4 in\n        \'man proc\' == position 1 in here).\n        The return value is cached in case oneshot() ctx manager is\n        in use.\n        '
        data = bcat('%s/%s/stat' % (self._procfs_path, self.pid))
        rpar = data.rfind(b')')
        name = data[data.find(b'(') + 1:rpar]
        fields = data[rpar + 2:].split()
        ret = {}
        ret['name'] = name
        ret['status'] = fields[0]
        ret['ppid'] = fields[1]
        ret['ttynr'] = fields[4]
        ret['utime'] = fields[11]
        ret['stime'] = fields[12]
        ret['children_utime'] = fields[13]
        ret['children_stime'] = fields[14]
        ret['create_time'] = fields[19]
        ret['cpu_num'] = fields[36]
        ret['blkio_ticks'] = fields[39]
        return ret

    @wrap_exceptions
    @memoize_when_activated
    def _read_status_file(self):
        if False:
            print('Hello World!')
        'Read /proc/{pid}/stat file and return its content.\n        The return value is cached in case oneshot() ctx manager is\n        in use.\n        '
        with open_binary('%s/%s/status' % (self._procfs_path, self.pid)) as f:
            return f.read()

    @wrap_exceptions
    @memoize_when_activated
    def _read_smaps_file(self):
        if False:
            while True:
                i = 10
        with open_binary('%s/%s/smaps' % (self._procfs_path, self.pid)) as f:
            return f.read().strip()

    def oneshot_enter(self):
        if False:
            i = 10
            return i + 15
        self._parse_stat_file.cache_activate(self)
        self._read_status_file.cache_activate(self)
        self._read_smaps_file.cache_activate(self)

    def oneshot_exit(self):
        if False:
            for i in range(10):
                print('nop')
        self._parse_stat_file.cache_deactivate(self)
        self._read_status_file.cache_deactivate(self)
        self._read_smaps_file.cache_deactivate(self)

    @wrap_exceptions
    def name(self):
        if False:
            for i in range(10):
                print('nop')
        name = self._parse_stat_file()['name']
        if PY3:
            name = decode(name)
        return name

    @wrap_exceptions
    def exe(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            return readlink('%s/%s/exe' % (self._procfs_path, self.pid))
        except (FileNotFoundError, ProcessLookupError):
            self._raise_if_zombie()
            if os.path.lexists('%s/%s' % (self._procfs_path, self.pid)):
                return ''
            raise

    @wrap_exceptions
    def cmdline(self):
        if False:
            while True:
                i = 10
        with open_text('%s/%s/cmdline' % (self._procfs_path, self.pid)) as f:
            data = f.read()
        if not data:
            self._raise_if_zombie()
            return []
        sep = '\x00' if data.endswith('\x00') else ' '
        if data.endswith(sep):
            data = data[:-1]
        cmdline = data.split(sep)
        if sep == '\x00' and len(cmdline) == 1 and (' ' in data):
            cmdline = data.split(' ')
        return cmdline

    @wrap_exceptions
    def environ(self):
        if False:
            print('Hello World!')
        with open_text('%s/%s/environ' % (self._procfs_path, self.pid)) as f:
            data = f.read()
        return parse_environ_block(data)

    @wrap_exceptions
    def terminal(self):
        if False:
            print('Hello World!')
        tty_nr = int(self._parse_stat_file()['ttynr'])
        tmap = _psposix.get_terminal_map()
        try:
            return tmap[tty_nr]
        except KeyError:
            return None
    if os.path.exists('/proc/%s/io' % os.getpid()):

        @wrap_exceptions
        def io_counters(self):
            if False:
                print('Hello World!')
            fname = '%s/%s/io' % (self._procfs_path, self.pid)
            fields = {}
            with open_binary(fname) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            (name, value) = line.split(b': ')
                        except ValueError:
                            continue
                        else:
                            fields[name] = int(value)
            if not fields:
                raise RuntimeError('%s file was empty' % fname)
            try:
                return pio(fields[b'syscr'], fields[b'syscw'], fields[b'read_bytes'], fields[b'write_bytes'], fields[b'rchar'], fields[b'wchar'])
            except KeyError as err:
                raise ValueError('%r field was not found in %s; found fields are %r' % (err.args[0], fname, fields))

    @wrap_exceptions
    def cpu_times(self):
        if False:
            print('Hello World!')
        values = self._parse_stat_file()
        utime = float(values['utime']) / CLOCK_TICKS
        stime = float(values['stime']) / CLOCK_TICKS
        children_utime = float(values['children_utime']) / CLOCK_TICKS
        children_stime = float(values['children_stime']) / CLOCK_TICKS
        iowait = float(values['blkio_ticks']) / CLOCK_TICKS
        return pcputimes(utime, stime, children_utime, children_stime, iowait)

    @wrap_exceptions
    def cpu_num(self):
        if False:
            i = 10
            return i + 15
        'What CPU the process is on.'
        return int(self._parse_stat_file()['cpu_num'])

    @wrap_exceptions
    def wait(self, timeout=None):
        if False:
            return 10
        return _psposix.wait_pid(self.pid, timeout, self._name)

    @wrap_exceptions
    def create_time(self):
        if False:
            for i in range(10):
                print('nop')
        ctime = float(self._parse_stat_file()['create_time'])
        bt = BOOT_TIME or boot_time()
        return ctime / CLOCK_TICKS + bt

    @wrap_exceptions
    def memory_info(self):
        if False:
            print('Hello World!')
        with open_binary('%s/%s/statm' % (self._procfs_path, self.pid)) as f:
            (vms, rss, shared, text, lib, data, dirty) = (int(x) * PAGESIZE for x in f.readline().split()[:7])
        return pmem(rss, vms, shared, text, lib, data, dirty)
    if HAS_PROC_SMAPS_ROLLUP or HAS_PROC_SMAPS:

        def _parse_smaps_rollup(self):
            if False:
                for i in range(10):
                    print('nop')
            uss = pss = swap = 0
            with open_binary('{}/{}/smaps_rollup'.format(self._procfs_path, self.pid)) as f:
                for line in f:
                    if line.startswith(b'Private_'):
                        uss += int(line.split()[1]) * 1024
                    elif line.startswith(b'Pss:'):
                        pss = int(line.split()[1]) * 1024
                    elif line.startswith(b'Swap:'):
                        swap = int(line.split()[1]) * 1024
            return (uss, pss, swap)

        @wrap_exceptions
        def _parse_smaps(self, _private_re=re.compile(b'\\nPrivate.*:\\s+(\\d+)'), _pss_re=re.compile(b'\\nPss\\:\\s+(\\d+)'), _swap_re=re.compile(b'\\nSwap\\:\\s+(\\d+)')):
            if False:
                for i in range(10):
                    print('nop')
            smaps_data = self._read_smaps_file()
            uss = sum(map(int, _private_re.findall(smaps_data))) * 1024
            pss = sum(map(int, _pss_re.findall(smaps_data))) * 1024
            swap = sum(map(int, _swap_re.findall(smaps_data))) * 1024
            return (uss, pss, swap)

        @wrap_exceptions
        def memory_full_info(self):
            if False:
                return 10
            if HAS_PROC_SMAPS_ROLLUP:
                try:
                    (uss, pss, swap) = self._parse_smaps_rollup()
                except (ProcessLookupError, FileNotFoundError) as err:
                    debug('ignore %r for pid %s and retry using /proc/pid/smaps' % (err, self.pid))
                    (uss, pss, swap) = self._parse_smaps()
            else:
                (uss, pss, swap) = self._parse_smaps()
            basic_mem = self.memory_info()
            return pfullmem(*basic_mem + (uss, pss, swap))
    else:
        memory_full_info = memory_info
    if HAS_PROC_SMAPS:

        @wrap_exceptions
        def memory_maps(self):
            if False:
                return 10
            "Return process's mapped memory regions as a list of named\n            tuples. Fields are explained in 'man proc'; here is an updated\n            (Apr 2012) version: http://goo.gl/fmebo.\n\n            /proc/{PID}/smaps does not exist on kernels < 2.6.14 or if\n            CONFIG_MMU kernel configuration option is not enabled.\n            "

            def get_blocks(lines, current_block):
                if False:
                    for i in range(10):
                        print('nop')
                data = {}
                for line in lines:
                    fields = line.split(None, 5)
                    if not fields[0].endswith(b':'):
                        yield (current_block.pop(), data)
                        current_block.append(line)
                    else:
                        try:
                            data[fields[0]] = int(fields[1]) * 1024
                        except ValueError:
                            if fields[0].startswith(b'VmFlags:'):
                                continue
                            else:
                                raise ValueError("don't know how to interpret line %r" % line)
                yield (current_block.pop(), data)
            data = self._read_smaps_file()
            if not data:
                self._raise_if_zombie()
                return []
            lines = data.split(b'\n')
            ls = []
            first_line = lines.pop(0)
            current_block = [first_line]
            for (header, data) in get_blocks(lines, current_block):
                hfields = header.split(None, 5)
                try:
                    (addr, perms, offset, dev, inode, path) = hfields
                except ValueError:
                    (addr, perms, offset, dev, inode, path) = hfields + ['']
                if not path:
                    path = '[anon]'
                else:
                    if PY3:
                        path = decode(path)
                    path = path.strip()
                    if path.endswith(' (deleted)') and (not path_exists_strict(path)):
                        path = path[:-10]
                ls.append((decode(addr), decode(perms), path, data.get(b'Rss:', 0), data.get(b'Size:', 0), data.get(b'Pss:', 0), data.get(b'Shared_Clean:', 0), data.get(b'Shared_Dirty:', 0), data.get(b'Private_Clean:', 0), data.get(b'Private_Dirty:', 0), data.get(b'Referenced:', 0), data.get(b'Anonymous:', 0), data.get(b'Swap:', 0)))
            return ls

    @wrap_exceptions
    def cwd(self):
        if False:
            print('Hello World!')
        return readlink('%s/%s/cwd' % (self._procfs_path, self.pid))

    @wrap_exceptions
    def num_ctx_switches(self, _ctxsw_re=re.compile(b'ctxt_switches:\\t(\\d+)')):
        if False:
            print('Hello World!')
        data = self._read_status_file()
        ctxsw = _ctxsw_re.findall(data)
        if not ctxsw:
            raise NotImplementedError("'voluntary_ctxt_switches' and 'nonvoluntary_ctxt_switches'lines were not found in %s/%s/status; the kernel is probably older than 2.6.23" % (self._procfs_path, self.pid))
        else:
            return _common.pctxsw(int(ctxsw[0]), int(ctxsw[1]))

    @wrap_exceptions
    def num_threads(self, _num_threads_re=re.compile(b'Threads:\\t(\\d+)')):
        if False:
            for i in range(10):
                print('nop')
        data = self._read_status_file()
        return int(_num_threads_re.findall(data)[0])

    @wrap_exceptions
    def threads(self):
        if False:
            for i in range(10):
                print('nop')
        thread_ids = os.listdir('%s/%s/task' % (self._procfs_path, self.pid))
        thread_ids.sort()
        retlist = []
        hit_enoent = False
        for thread_id in thread_ids:
            fname = '%s/%s/task/%s/stat' % (self._procfs_path, self.pid, thread_id)
            try:
                with open_binary(fname) as f:
                    st = f.read().strip()
            except (FileNotFoundError, ProcessLookupError):
                hit_enoent = True
                continue
            st = st[st.find(b')') + 2:]
            values = st.split(b' ')
            utime = float(values[11]) / CLOCK_TICKS
            stime = float(values[12]) / CLOCK_TICKS
            ntuple = _common.pthread(int(thread_id), utime, stime)
            retlist.append(ntuple)
        if hit_enoent:
            self._raise_if_not_alive()
        return retlist

    @wrap_exceptions
    def nice_get(self):
        if False:
            for i in range(10):
                print('nop')
        return cext_posix.getpriority(self.pid)

    @wrap_exceptions
    def nice_set(self, value):
        if False:
            i = 10
            return i + 15
        return cext_posix.setpriority(self.pid, value)
    if HAS_CPU_AFFINITY:

        @wrap_exceptions
        def cpu_affinity_get(self):
            if False:
                return 10
            return cext.proc_cpu_affinity_get(self.pid)

        def _get_eligible_cpus(self, _re=re.compile(b'Cpus_allowed_list:\\t(\\d+)-(\\d+)')):
            if False:
                print('Hello World!')
            data = self._read_status_file()
            match = _re.findall(data)
            if match:
                return list(range(int(match[0][0]), int(match[0][1]) + 1))
            else:
                return list(range(len(per_cpu_times())))

        @wrap_exceptions
        def cpu_affinity_set(self, cpus):
            if False:
                return 10
            try:
                cext.proc_cpu_affinity_set(self.pid, cpus)
            except (OSError, ValueError) as err:
                if isinstance(err, ValueError) or err.errno == errno.EINVAL:
                    eligible_cpus = self._get_eligible_cpus()
                    all_cpus = tuple(range(len(per_cpu_times())))
                    for cpu in cpus:
                        if cpu not in all_cpus:
                            raise ValueError('invalid CPU number %r; choose between %s' % (cpu, eligible_cpus))
                        if cpu not in eligible_cpus:
                            raise ValueError('CPU number %r is not eligible; choose between %s' % (cpu, eligible_cpus))
                raise
    if HAS_PROC_IO_PRIORITY:

        @wrap_exceptions
        def ionice_get(self):
            if False:
                while True:
                    i = 10
            (ioclass, value) = cext.proc_ioprio_get(self.pid)
            if enum is not None:
                ioclass = IOPriority(ioclass)
            return _common.pionice(ioclass, value)

        @wrap_exceptions
        def ionice_set(self, ioclass, value):
            if False:
                return 10
            if value is None:
                value = 0
            if value and ioclass in (IOPRIO_CLASS_IDLE, IOPRIO_CLASS_NONE):
                raise ValueError('%r ioclass accepts no value' % ioclass)
            if value < 0 or value > 7:
                msg = 'value not in 0-7 range'
                raise ValueError(msg)
            return cext.proc_ioprio_set(self.pid, ioclass, value)
    if prlimit is not None:

        @wrap_exceptions
        def rlimit(self, resource_, limits=None):
            if False:
                while True:
                    i = 10
            if self.pid == 0:
                msg = "can't use prlimit() against PID 0 process"
                raise ValueError(msg)
            try:
                if limits is None:
                    return prlimit(self.pid, resource_)
                else:
                    if len(limits) != 2:
                        raise ValueError('second argument must be a (soft, hard) tuple, got %s' % repr(limits))
                    prlimit(self.pid, resource_, limits)
            except OSError as err:
                if err.errno == errno.ENOSYS:
                    self._raise_if_zombie()
                raise

    @wrap_exceptions
    def status(self):
        if False:
            i = 10
            return i + 15
        letter = self._parse_stat_file()['status']
        if PY3:
            letter = letter.decode()
        return PROC_STATUSES.get(letter, '?')

    @wrap_exceptions
    def open_files(self):
        if False:
            return 10
        retlist = []
        files = os.listdir('%s/%s/fd' % (self._procfs_path, self.pid))
        hit_enoent = False
        for fd in files:
            file = '%s/%s/fd/%s' % (self._procfs_path, self.pid, fd)
            try:
                path = readlink(file)
            except (FileNotFoundError, ProcessLookupError):
                hit_enoent = True
                continue
            except OSError as err:
                if err.errno == errno.EINVAL:
                    continue
                if err.errno == errno.ENAMETOOLONG:
                    debug(err)
                    continue
                raise
            else:
                if path.startswith('/') and isfile_strict(path):
                    file = '%s/%s/fdinfo/%s' % (self._procfs_path, self.pid, fd)
                    try:
                        with open_binary(file) as f:
                            pos = int(f.readline().split()[1])
                            flags = int(f.readline().split()[1], 8)
                    except (FileNotFoundError, ProcessLookupError):
                        hit_enoent = True
                    else:
                        mode = file_flags_to_mode(flags)
                        ntuple = popenfile(path, int(fd), int(pos), mode, flags)
                        retlist.append(ntuple)
        if hit_enoent:
            self._raise_if_not_alive()
        return retlist

    @wrap_exceptions
    def connections(self, kind='inet'):
        if False:
            print('Hello World!')
        ret = _connections.retrieve(kind, self.pid)
        self._raise_if_not_alive()
        return ret

    @wrap_exceptions
    def num_fds(self):
        if False:
            i = 10
            return i + 15
        return len(os.listdir('%s/%s/fd' % (self._procfs_path, self.pid)))

    @wrap_exceptions
    def ppid(self):
        if False:
            i = 10
            return i + 15
        return int(self._parse_stat_file()['ppid'])

    @wrap_exceptions
    def uids(self, _uids_re=re.compile(b'Uid:\\t(\\d+)\\t(\\d+)\\t(\\d+)')):
        if False:
            print('Hello World!')
        data = self._read_status_file()
        (real, effective, saved) = _uids_re.findall(data)[0]
        return _common.puids(int(real), int(effective), int(saved))

    @wrap_exceptions
    def gids(self, _gids_re=re.compile(b'Gid:\\t(\\d+)\\t(\\d+)\\t(\\d+)')):
        if False:
            return 10
        data = self._read_status_file()
        (real, effective, saved) = _gids_re.findall(data)[0]
        return _common.pgids(int(real), int(effective), int(saved))