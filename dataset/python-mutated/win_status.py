"""
Module for returning various status data about a minion.
These data can be useful for compiling into stats later,
or for problem solving if your minion is having problems.

.. versionadded:: 0.12.0

:depends:  - wmi
"""
import ctypes
import datetime
import logging
import subprocess
import salt.utils.event
import salt.utils.platform
import salt.utils.stringutils
import salt.utils.win_pdh
from salt.modules.status import ping_master, time_
from salt.utils.functools import namespaced_function
from salt.utils.network import host_to_ips as _host_to_ips
log = logging.getLogger(__name__)
try:
    if salt.utils.platform.is_windows():
        import wmi
        import salt.utils.winapi
        HAS_WMI = True
    else:
        HAS_WMI = False
except ImportError:
    HAS_WMI = False
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
if salt.utils.platform.is_windows():
    ping_master = namespaced_function(ping_master, globals())
    time_ = namespaced_function(time_, globals())
__virtualname__ = 'status'

class SYSTEM_PERFORMANCE_INFORMATION(ctypes.Structure):
    _fields_ = [('IdleProcessTime', ctypes.c_int64), ('IoReadTransferCount', ctypes.c_int64), ('IoWriteTransferCount', ctypes.c_int64), ('IoOtherTransferCount', ctypes.c_int64), ('IoReadOperationCount', ctypes.c_ulong), ('IoWriteOperationCount', ctypes.c_ulong), ('IoOtherOperationCount', ctypes.c_ulong), ('AvailablePages', ctypes.c_ulong), ('CommittedPages', ctypes.c_ulong), ('CommitLimit', ctypes.c_ulong), ('PeakCommitment', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong), ('CopyOnWriteCount', ctypes.c_ulong), ('TransitionCount', ctypes.c_ulong), ('CacheTransitionCount', ctypes.c_ulong), ('DemandZeroCount', ctypes.c_ulong), ('PageReadCount', ctypes.c_ulong), ('PageReadIoCount', ctypes.c_ulong), ('CacheReadCount', ctypes.c_ulong), ('CacheIoCount', ctypes.c_ulong), ('DirtyPagesWriteCount', ctypes.c_ulong), ('DirtyWriteIoCount', ctypes.c_ulong), ('MappedPagesWriteCount', ctypes.c_ulong), ('MappedWriteIoCount', ctypes.c_ulong), ('PagedPoolPages', ctypes.c_ulong), ('NonPagedPoolPages', ctypes.c_ulong), ('PagedPoolAllocs', ctypes.c_ulong), ('PagedPoolFrees', ctypes.c_ulong), ('NonPagedPoolAllocs', ctypes.c_ulong), ('NonPagedPoolFrees', ctypes.c_ulong), ('FreeSystemPtes', ctypes.c_ulong), ('ResidentSystemCodePage', ctypes.c_ulong), ('TotalSystemDriverPages', ctypes.c_ulong), ('TotalSystemCodePages', ctypes.c_ulong), ('NonPagedPoolLookasideHits', ctypes.c_ulong), ('PagedPoolLookasideHits', ctypes.c_ulong), ('AvailablePagedPoolPages', ctypes.c_ulong), ('ResidentSystemCachePage', ctypes.c_ulong), ('ResidentPagedPoolPage', ctypes.c_ulong), ('ResidentSystemDriverPage', ctypes.c_ulong), ('CcFastReadNoWait', ctypes.c_ulong), ('CcFastReadWait', ctypes.c_ulong), ('CcFastReadResourceMiss', ctypes.c_ulong), ('CcFastReadNotPossible', ctypes.c_ulong), ('CcFastMdlReadNoWait', ctypes.c_ulong), ('CcFastMdlReadWait', ctypes.c_ulong), ('CcFastMdlReadResourceMiss', ctypes.c_ulong), ('CcFastMdlReadNotPossible', ctypes.c_ulong), ('CcMapDataNoWait', ctypes.c_ulong), ('CcMapDataWait', ctypes.c_ulong), ('CcMapDataNoWaitMiss', ctypes.c_ulong), ('CcMapDataWaitMiss', ctypes.c_ulong), ('CcPinMappedDataCount', ctypes.c_ulong), ('CcPinReadNoWait', ctypes.c_ulong), ('CcPinReadWait', ctypes.c_ulong), ('CcPinReadNoWaitMiss', ctypes.c_ulong), ('CcPinReadWaitMiss', ctypes.c_ulong), ('CcCopyReadNoWait', ctypes.c_ulong), ('CcCopyReadWait', ctypes.c_ulong), ('CcCopyReadNoWaitMiss', ctypes.c_ulong), ('CcCopyReadWaitMiss', ctypes.c_ulong), ('CcMdlReadNoWait', ctypes.c_ulong), ('CcMdlReadWait', ctypes.c_ulong), ('CcMdlReadNoWaitMiss', ctypes.c_ulong), ('CcMdlReadWaitMiss', ctypes.c_ulong), ('CcReadAheadIos', ctypes.c_ulong), ('CcLazyWriteIos', ctypes.c_ulong), ('CcLazyWritePages', ctypes.c_ulong), ('CcDataFlushes', ctypes.c_ulong), ('CcDataPages', ctypes.c_ulong), ('ContextSwitches', ctypes.c_ulong), ('FirstLevelTbFills', ctypes.c_ulong), ('SecondLevelTbFills', ctypes.c_ulong), ('SystemCalls', ctypes.c_ulong), ('CcTotalDirtyPages', ctypes.c_ulonglong), ('CcDirtyPagesThreshold', ctypes.c_ulonglong), ('ResidentAvailablePages', ctypes.c_longlong), ('SharedCommittedPages', ctypes.c_ulonglong)]

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    Only works on Windows systems with WMI and WinAPI\n    '
    if not salt.utils.platform.is_windows():
        return (False, 'win_status.py: Requires Windows')
    if not HAS_WMI:
        return (False, 'win_status.py: Requires WMI and WinAPI')
    if not HAS_PSUTIL:
        return (False, 'win_status.py: Requires psutil')
    global ping_master, time_
    return __virtualname__
__func_alias__ = {'time_': 'time'}

def cpustats():
    if False:
        while True:
            i = 10
    '\n    Return information about the CPU.\n\n    Returns\n        dict: A dictionary containing information about the CPU stats\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt * status.cpustats\n    '
    (user, system, idle, interrupt, dpc) = psutil.cpu_times()
    cpu = {'user': user, 'system': system, 'idle': idle, 'irq': interrupt, 'dpc': dpc}
    (ctx_switches, interrupts, soft_interrupts, sys_calls) = psutil.cpu_stats()
    intr = {'irqs': {'irqs': [], 'total': interrupts}}
    soft_irq = {'softirqs': [], 'total': soft_interrupts}
    return {'btime': psutil.boot_time(), 'cpu': cpu, 'ctxt': ctx_switches, 'intr': intr, 'processes': len(psutil.pids()), 'softirq': soft_irq, 'syscalls': sys_calls}

def meminfo():
    if False:
        print('Hello World!')
    '\n    Return information about physical and virtual memory on the system\n\n    Returns:\n        dict: A dictionary of information about memory on the system\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt * status.meminfo\n    '
    (vm_total, vm_available, vm_percent, vm_used, vm_free) = psutil.virtual_memory()
    (swp_total, swp_used, swp_free, swp_percent, _, _) = psutil.swap_memory()

    def get_unit_value(memory):
        if False:
            i = 10
            return i + 15
        symbols = ('K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
        prefix = {}
        for (i, s) in enumerate(symbols):
            prefix[s] = 1 << (i + 1) * 10
        for s in reversed(symbols):
            if memory >= prefix[s]:
                value = float(memory) / prefix[s]
                return {'unit': s, 'value': value}
        return {'unit': 'B', 'value': memory}
    return {'VmallocTotal': get_unit_value(vm_total), 'VmallocUsed': get_unit_value(vm_used), 'VmallocFree': get_unit_value(vm_free), 'VmallocAvail': get_unit_value(vm_available), 'SwapTotal': get_unit_value(swp_total), 'SwapUsed': get_unit_value(swp_used), 'SwapFree': get_unit_value(swp_free)}

def vmstats():
    if False:
        for i in range(10):
            print('nop')
    '\n    Return information about the virtual memory on the machine\n\n    Returns:\n        dict: A dictionary of virtual memory stats\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt * status.vmstats\n    '
    spi = SYSTEM_PERFORMANCE_INFORMATION()
    retlen = ctypes.c_ulong()
    ctypes.windll.ntdll.NtQuerySystemInformation(2, ctypes.byref(spi), ctypes.sizeof(spi), ctypes.byref(retlen))
    ret = {}
    for field in spi._fields_:
        ret.update({field[0]: getattr(spi, field[0])})
    return ret

def loadavg():
    if False:
        i = 10
        return i + 15
    '\n    Returns counter information related to the load of the machine\n\n    Returns:\n        dict: A dictionary of counters\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt * status.loadavg\n    '
    counter_list = [('Memory', None, 'Available Bytes'), ('Memory', None, 'Pages/sec'), ('Paging File', '*', '% Usage'), ('Processor', '*', '% Processor Time'), ('Processor', '*', 'DPCs Queued/sec'), ('Processor', '*', '% Privileged Time'), ('Processor', '*', '% User Time'), ('Processor', '*', '% DPC Time'), ('Processor', '*', '% Interrupt Time'), ('Server', None, 'Work Item Shortages'), ('Server Work Queues', '*', 'Queue Length'), ('System', None, 'Processor Queue Length'), ('System', None, 'Context Switches/sec')]
    return salt.utils.win_pdh.get_counters(counter_list=counter_list)

def cpuload():
    if False:
        i = 10
        return i + 15
    "\n    .. versionadded:: 2015.8.0\n\n    Return the processor load as a percentage\n\n    CLI Example:\n\n    .. code-block:: bash\n\n       salt '*' status.cpuload\n    "
    return psutil.cpu_percent()

def diskusage(human_readable=False, path=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    .. versionadded:: 2015.8.0\n\n    Return the disk usage for this minion\n\n    human_readable : False\n        If ``True``, usage will be in KB/MB/GB etc.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' status.diskusage path=c:/salt\n    "
    if not path:
        path = 'c:/'
    disk_stats = psutil.disk_usage(path)
    total_val = disk_stats.total
    used_val = disk_stats.used
    free_val = disk_stats.free
    percent = disk_stats.percent
    if human_readable:
        total_val = _byte_calc(total_val)
        used_val = _byte_calc(used_val)
        free_val = _byte_calc(free_val)
    return {'total': total_val, 'used': used_val, 'free': free_val, 'percent': percent}

def procs(count=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return the process data\n\n    count : False\n        If ``True``, this function will simply return the number of processes.\n\n        .. versionadded:: 2015.8.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' status.procs\n        salt '*' status.procs count\n    "
    with salt.utils.winapi.Com():
        wmi_obj = wmi.WMI()
        processes = wmi_obj.win32_process()
    if count:
        return len(processes)
    process_info = {}
    for proc in processes:
        process_info[proc.ProcessId] = _get_process_info(proc)
    return process_info

def saltmem(human_readable=False):
    if False:
        return 10
    "\n    .. versionadded:: 2015.8.0\n\n    Returns the amount of memory that salt is using\n\n    human_readable : False\n        return the value in a nicely formatted number\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' status.saltmem\n        salt '*' status.saltmem human_readable=True\n    "
    p = psutil.Process()
    with p.oneshot():
        mem = p.memory_info().rss
    if human_readable:
        return _byte_calc(mem)
    return mem

def uptime(human_readable=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    .. versionadded:: 2015.8.0\n\n    Return the system uptime for the machine\n\n    Args:\n\n        human_readable (bool):\n            Return uptime in human readable format if ``True``, otherwise\n            return seconds. Default is ``False``\n\n            .. note::\n                Human readable format is ``days, hours:min:sec``. Days will only\n                be displayed if more than 0\n\n    Returns:\n        str:\n            The uptime in seconds or human readable format depending on the\n            value of ``human_readable``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' status.uptime\n        salt '*' status.uptime human_readable=True\n    "
    startup_time = datetime.datetime.fromtimestamp(psutil.boot_time())
    uptime = datetime.datetime.now() - startup_time
    return str(uptime) if human_readable else uptime.total_seconds()

def _get_process_info(proc):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return  process information\n    '
    cmd = salt.utils.stringutils.to_unicode(proc.CommandLine or '')
    name = salt.utils.stringutils.to_unicode(proc.Name)
    info = dict(cmd=cmd, name=name, **_get_process_owner(proc))
    return info

def _get_process_owner(process):
    if False:
        print('Hello World!')
    owner = {}
    (domain, error_code, user) = (None, None, None)
    try:
        (domain, error_code, user) = process.GetOwner()
        owner['user'] = salt.utils.stringutils.to_unicode(user)
        owner['user_domain'] = salt.utils.stringutils.to_unicode(domain)
    except Exception as exc:
        pass
    if not error_code and all((user, domain)):
        owner['user'] = salt.utils.stringutils.to_unicode(user)
        owner['user_domain'] = salt.utils.stringutils.to_unicode(domain)
    elif process.ProcessId in [0, 4] and error_code == 2:
        owner['user'] = 'SYSTEM'
        owner['user_domain'] = 'NT AUTHORITY'
    else:
        log.warning("Error getting owner of process; PID='%s'; Error: %s", process.ProcessId, error_code)
    return owner

def _byte_calc(val):
    if False:
        i = 10
        return i + 15
    if val < 1024:
        tstr = str(val) + 'B'
    elif val < 1038336:
        tstr = str(val / 1024) + 'KB'
    elif val < 1073741824:
        tstr = str(val / 1038336) + 'MB'
    elif val < 1099511627776:
        tstr = str(val / 1073741824) + 'GB'
    else:
        tstr = str(val / 1099511627776) + 'TB'
    return tstr

def master(master=None, connected=True):
    if False:
        for i in range(10):
            print('nop')
    "\n    .. versionadded:: 2015.5.0\n\n    Fire an event if the minion gets disconnected from its master. This\n    function is meant to be run via a scheduled job from the minion. If\n    master_ip is an FQDN/Hostname, is must be resolvable to a valid IPv4\n    address.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' status.master\n    "

    def _win_remotes_on(port):
        if False:
            return 10
        "\n        Windows specific helper function.\n        Returns set of ipv4 host addresses of remote established connections\n        on local or remote tcp port.\n\n        Parses output of shell 'netstat' to get connections\n\n        PS C:> netstat -n -p TCP\n\n        Active Connections\n\n          Proto  Local Address          Foreign Address        State\n          TCP    10.1.1.26:3389         10.1.1.1:4505          ESTABLISHED\n          TCP    10.1.1.26:56862        10.1.1.10:49155        TIME_WAIT\n          TCP    10.1.1.26:56868        169.254.169.254:80     CLOSE_WAIT\n          TCP    127.0.0.1:49197        127.0.0.1:49198        ESTABLISHED\n          TCP    127.0.0.1:49198        127.0.0.1:49197        ESTABLISHED\n        "
        remotes = set()
        try:
            data = subprocess.check_output(['netstat', '-n', '-p', 'TCP'])
        except subprocess.CalledProcessError:
            log.error('Failed netstat')
            raise
        lines = salt.utils.stringutils.to_unicode(data).split('\n')
        for line in lines:
            if 'ESTABLISHED' not in line:
                continue
            chunks = line.split()
            (remote_host, remote_port) = chunks[2].rsplit(':', 1)
            if int(remote_port) != port:
                continue
            remotes.add(remote_host)
        return remotes
    port = 4505
    master_ips = None
    if master:
        master_ips = _host_to_ips(master)
    if not master_ips:
        return
    if __salt__['config.get']('publish_port') != '':
        port = int(__salt__['config.get']('publish_port'))
    master_connection_status = False
    connected_ips = _win_remotes_on(port)
    for master_ip in master_ips:
        if master_ip in connected_ips:
            master_connection_status = True
            break
    if master_connection_status is not connected:
        with salt.utils.event.get_event('minion', opts=__opts__, listen=False) as event_bus:
            if master_connection_status:
                event_bus.fire_event({'master': master}, salt.minion.master_event(type='connected'))
            else:
                event_bus.fire_event({'master': master}, salt.minion.master_event(type='disconnected'))
    return master_connection_status