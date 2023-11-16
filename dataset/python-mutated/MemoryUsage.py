""" Tools for tracing memory usage at compiled time.

"""
from nuitka.containers.OrderedDicts import OrderedDict
from nuitka.Tracing import memory_logger, printLine
from .Utils import isMacOS, isWin32Windows

def getOwnProcessMemoryUsage():
    if False:
        for i in range(10):
            print('nop')
    'Memory usage of own process in bytes.'
    if isWin32Windows():
        import ctypes.wintypes

        class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
            _fields_ = [('cb', ctypes.wintypes.DWORD), ('PageFaultCount', ctypes.wintypes.DWORD), ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t), ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t), ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t), ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t), ('PrivateUsage', ctypes.c_size_t)]
        GetProcessMemoryInfo = ctypes.windll.psapi.GetProcessMemoryInfo
        GetProcessMemoryInfo.argtypes = (ctypes.wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX), ctypes.wintypes.DWORD)
        GetProcessMemoryInfo.restype = ctypes.wintypes.BOOL
        counters = PROCESS_MEMORY_COUNTERS_EX()
        rv = GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), ctypes.sizeof(counters))
        if not rv:
            raise ctypes.WinError()
        return counters.PrivateUsage
    else:
        import resource
        if isMacOS():
            factor = 1
        else:
            factor = 1024
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * factor

def getHumanReadableProcessMemoryUsage():
    if False:
        return 10
    return formatMemoryUsageValue(getOwnProcessMemoryUsage())

class MemoryWatch(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.start = getOwnProcessMemoryUsage()
        self.stop = None

    def finish(self, message):
        if False:
            print('Hello World!')
        self.stop = getOwnProcessMemoryUsage()
        _logMemoryInfo(message, self.value())

    def asStr(self):
        if False:
            i = 10
            return i + 15
        return formatMemoryUsageValue(self.value())

    def value(self):
        if False:
            print('Hello World!')
        return self.stop - self.start
_memory_infos = OrderedDict()

def getMemoryInfos():
    if False:
        while True:
            i = 10
    return _memory_infos

def collectMemoryUsageValue(memory_usage_name):
    if False:
        i = 10
        return i + 15
    assert memory_usage_name not in _memory_infos
    _memory_infos[memory_usage_name] = getOwnProcessMemoryUsage()
    return _memory_infos[memory_usage_name]

def formatMemoryUsageValue(value):
    if False:
        return 10
    if abs(value) < 1024 * 1014:
        return '%.2f KB (%d bytes)' % (value / 1024.0, value)
    elif abs(value) < 1024 * 1014 * 1024:
        return '%.2f MB (%d bytes)' % (value / (1024 * 1024.0), value)
    elif abs(value) < 1024 * 1014 * 1024 * 1024:
        return '%.2f GB (%d bytes)' % (value / (1024 * 1024 * 1024.0), value)
    else:
        return '%d bytes' % value

def _logMemoryInfo(message, memory_usage):
    if False:
        print('Hello World!')
    if message:
        memory_logger.info('%s: %s' % (message, formatMemoryUsageValue(memory_usage)))

def reportMemoryUsage(identifier, message):
    if False:
        i = 10
        return i + 15
    memory_usage = collectMemoryUsageValue(identifier)
    _logMemoryInfo(message, memory_usage)

def startMemoryTracing():
    if False:
        print('Hello World!')
    try:
        import tracemalloc
    except ImportError:
        pass
    else:
        tracemalloc.start()

def showMemoryTrace():
    if False:
        while True:
            i = 10
    try:
        import tracemalloc
    except ImportError:
        pass
    else:
        snapshot = tracemalloc.take_snapshot()
        stats = snapshot.statistics('lineno')
        printLine('Top 50 memory allocations:')
        for (count, stat) in enumerate(stats):
            if count == 50:
                break
            printLine(stat)