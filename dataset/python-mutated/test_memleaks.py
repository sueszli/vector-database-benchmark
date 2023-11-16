"""Tests for detecting function memory leaks (typically the ones
implemented in C). It does so by calling a function many times and
checking whether process memory usage keeps increasing between
calls or over time.
Note that this may produce false positives (especially on Windows
for some reason).
PyPy appears to be completely unstable for this framework, probably
because of how its JIT handles memory, so tests are skipped.
"""
from __future__ import print_function
import functools
import os
import platform
import unittest
import psutil
import psutil._common
from psutil import LINUX
from psutil import MACOS
from psutil import OPENBSD
from psutil import POSIX
from psutil import SUNOS
from psutil import WINDOWS
from psutil._compat import ProcessLookupError
from psutil._compat import super
from psutil.tests import HAS_CPU_AFFINITY
from psutil.tests import HAS_CPU_FREQ
from psutil.tests import HAS_ENVIRON
from psutil.tests import HAS_IONICE
from psutil.tests import HAS_MEMORY_MAPS
from psutil.tests import HAS_NET_IO_COUNTERS
from psutil.tests import HAS_PROC_CPU_NUM
from psutil.tests import HAS_PROC_IO_COUNTERS
from psutil.tests import HAS_RLIMIT
from psutil.tests import HAS_SENSORS_BATTERY
from psutil.tests import HAS_SENSORS_FANS
from psutil.tests import HAS_SENSORS_TEMPERATURES
from psutil.tests import TestMemoryLeak
from psutil.tests import create_sockets
from psutil.tests import get_testfn
from psutil.tests import process_namespace
from psutil.tests import skip_on_access_denied
from psutil.tests import spawn_testproc
from psutil.tests import system_namespace
from psutil.tests import terminate
cext = psutil._psplatform.cext
thisproc = psutil.Process()
FEW_TIMES = 5

def fewtimes_if_linux():
    if False:
        print('Hello World!')
    'Decorator for those Linux functions which are implemented in pure\n    Python, and which we want to run faster.\n    '

    def decorator(fun):
        if False:
            i = 10
            return i + 15

        @functools.wraps(fun)
        def wrapper(self, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            if LINUX:
                before = self.__class__.times
                try:
                    self.__class__.times = FEW_TIMES
                    return fun(self, *args, **kwargs)
                finally:
                    self.__class__.times = before
            else:
                return fun(self, *args, **kwargs)
        return wrapper
    return decorator

class TestProcessObjectLeaks(TestMemoryLeak):
    """Test leaks of Process class methods."""
    proc = thisproc

    def test_coverage(self):
        if False:
            print('Hello World!')
        ns = process_namespace(None)
        ns.test_class_coverage(self, ns.getters + ns.setters)

    @fewtimes_if_linux()
    def test_name(self):
        if False:
            for i in range(10):
                print('nop')
        self.execute(self.proc.name)

    @fewtimes_if_linux()
    def test_cmdline(self):
        if False:
            print('Hello World!')
        self.execute(self.proc.cmdline)

    @fewtimes_if_linux()
    def test_exe(self):
        if False:
            i = 10
            return i + 15
        self.execute(self.proc.exe)

    @fewtimes_if_linux()
    def test_ppid(self):
        if False:
            for i in range(10):
                print('nop')
        self.execute(self.proc.ppid)

    @unittest.skipIf(not POSIX, 'POSIX only')
    @fewtimes_if_linux()
    def test_uids(self):
        if False:
            while True:
                i = 10
        self.execute(self.proc.uids)

    @unittest.skipIf(not POSIX, 'POSIX only')
    @fewtimes_if_linux()
    def test_gids(self):
        if False:
            return 10
        self.execute(self.proc.gids)

    @fewtimes_if_linux()
    def test_status(self):
        if False:
            return 10
        self.execute(self.proc.status)

    def test_nice(self):
        if False:
            while True:
                i = 10
        self.execute(self.proc.nice)

    def test_nice_set(self):
        if False:
            print('Hello World!')
        niceness = thisproc.nice()
        self.execute(lambda : self.proc.nice(niceness))

    @unittest.skipIf(not HAS_IONICE, 'not supported')
    def test_ionice(self):
        if False:
            while True:
                i = 10
        self.execute(self.proc.ionice)

    @unittest.skipIf(not HAS_IONICE, 'not supported')
    def test_ionice_set(self):
        if False:
            i = 10
            return i + 15
        if WINDOWS:
            value = thisproc.ionice()
            self.execute(lambda : self.proc.ionice(value))
        else:
            self.execute(lambda : self.proc.ionice(psutil.IOPRIO_CLASS_NONE))
            fun = functools.partial(cext.proc_ioprio_set, os.getpid(), -1, 0)
            self.execute_w_exc(OSError, fun)

    @unittest.skipIf(not HAS_PROC_IO_COUNTERS, 'not supported')
    @fewtimes_if_linux()
    def test_io_counters(self):
        if False:
            return 10
        self.execute(self.proc.io_counters)

    @unittest.skipIf(POSIX, 'worthless on POSIX')
    def test_username(self):
        if False:
            return 10
        psutil.Process().username()
        self.execute(self.proc.username)

    @fewtimes_if_linux()
    def test_create_time(self):
        if False:
            print('Hello World!')
        self.execute(self.proc.create_time)

    @fewtimes_if_linux()
    @skip_on_access_denied(only_if=OPENBSD)
    def test_num_threads(self):
        if False:
            while True:
                i = 10
        self.execute(self.proc.num_threads)

    @unittest.skipIf(not WINDOWS, 'WINDOWS only')
    def test_num_handles(self):
        if False:
            for i in range(10):
                print('nop')
        self.execute(self.proc.num_handles)

    @unittest.skipIf(not POSIX, 'POSIX only')
    @fewtimes_if_linux()
    def test_num_fds(self):
        if False:
            for i in range(10):
                print('nop')
        self.execute(self.proc.num_fds)

    @fewtimes_if_linux()
    def test_num_ctx_switches(self):
        if False:
            while True:
                i = 10
        self.execute(self.proc.num_ctx_switches)

    @fewtimes_if_linux()
    @skip_on_access_denied(only_if=OPENBSD)
    def test_threads(self):
        if False:
            while True:
                i = 10
        self.execute(self.proc.threads)

    @fewtimes_if_linux()
    def test_cpu_times(self):
        if False:
            for i in range(10):
                print('nop')
        self.execute(self.proc.cpu_times)

    @fewtimes_if_linux()
    @unittest.skipIf(not HAS_PROC_CPU_NUM, 'not supported')
    def test_cpu_num(self):
        if False:
            for i in range(10):
                print('nop')
        self.execute(self.proc.cpu_num)

    @fewtimes_if_linux()
    def test_memory_info(self):
        if False:
            i = 10
            return i + 15
        self.execute(self.proc.memory_info)

    @fewtimes_if_linux()
    def test_memory_full_info(self):
        if False:
            print('Hello World!')
        self.execute(self.proc.memory_full_info)

    @unittest.skipIf(not POSIX, 'POSIX only')
    @fewtimes_if_linux()
    def test_terminal(self):
        if False:
            return 10
        self.execute(self.proc.terminal)

    def test_resume(self):
        if False:
            for i in range(10):
                print('nop')
        times = FEW_TIMES if POSIX else self.times
        self.execute(self.proc.resume, times=times)

    @fewtimes_if_linux()
    def test_cwd(self):
        if False:
            print('Hello World!')
        self.execute(self.proc.cwd)

    @unittest.skipIf(not HAS_CPU_AFFINITY, 'not supported')
    def test_cpu_affinity(self):
        if False:
            while True:
                i = 10
        self.execute(self.proc.cpu_affinity)

    @unittest.skipIf(not HAS_CPU_AFFINITY, 'not supported')
    def test_cpu_affinity_set(self):
        if False:
            return 10
        affinity = thisproc.cpu_affinity()
        self.execute(lambda : self.proc.cpu_affinity(affinity))
        self.execute_w_exc(ValueError, lambda : self.proc.cpu_affinity([-1]))

    @fewtimes_if_linux()
    def test_open_files(self):
        if False:
            return 10
        with open(get_testfn(), 'w'):
            self.execute(self.proc.open_files)

    @unittest.skipIf(not HAS_MEMORY_MAPS, 'not supported')
    @fewtimes_if_linux()
    def test_memory_maps(self):
        if False:
            for i in range(10):
                print('nop')
        self.execute(self.proc.memory_maps)

    @unittest.skipIf(not LINUX, 'LINUX only')
    @unittest.skipIf(not HAS_RLIMIT, 'not supported')
    def test_rlimit(self):
        if False:
            for i in range(10):
                print('nop')
        self.execute(lambda : self.proc.rlimit(psutil.RLIMIT_NOFILE))

    @unittest.skipIf(not LINUX, 'LINUX only')
    @unittest.skipIf(not HAS_RLIMIT, 'not supported')
    def test_rlimit_set(self):
        if False:
            print('Hello World!')
        limit = thisproc.rlimit(psutil.RLIMIT_NOFILE)
        self.execute(lambda : self.proc.rlimit(psutil.RLIMIT_NOFILE, limit))
        self.execute_w_exc((OSError, ValueError), lambda : self.proc.rlimit(-1))

    @fewtimes_if_linux()
    @unittest.skipIf(WINDOWS, 'worthless on WINDOWS')
    def test_connections(self):
        if False:
            while True:
                i = 10
        with create_sockets():
            kind = 'inet' if SUNOS else 'all'
            self.execute(lambda : self.proc.connections(kind))

    @unittest.skipIf(not HAS_ENVIRON, 'not supported')
    def test_environ(self):
        if False:
            i = 10
            return i + 15
        self.execute(self.proc.environ)

    @unittest.skipIf(not WINDOWS, 'WINDOWS only')
    def test_proc_info(self):
        if False:
            i = 10
            return i + 15
        self.execute(lambda : cext.proc_info(os.getpid()))

class TestTerminatedProcessLeaks(TestProcessObjectLeaks):
    """Repeat the tests above looking for leaks occurring when dealing
    with terminated processes raising NoSuchProcess exception.
    The C functions are still invoked but will follow different code
    paths. We'll check those code paths.
    """

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        super().setUpClass()
        cls.subp = spawn_testproc()
        cls.proc = psutil.Process(cls.subp.pid)
        cls.proc.kill()
        cls.proc.wait()

    @classmethod
    def tearDownClass(cls):
        if False:
            for i in range(10):
                print('nop')
        super().tearDownClass()
        terminate(cls.subp)

    def call(self, fun):
        if False:
            print('Hello World!')
        try:
            fun()
        except psutil.NoSuchProcess:
            pass
    if WINDOWS:

        def test_kill(self):
            if False:
                return 10
            self.execute(self.proc.kill)

        def test_terminate(self):
            if False:
                print('Hello World!')
            self.execute(self.proc.terminate)

        def test_suspend(self):
            if False:
                print('Hello World!')
            self.execute(self.proc.suspend)

        def test_resume(self):
            if False:
                print('Hello World!')
            self.execute(self.proc.resume)

        def test_wait(self):
            if False:
                for i in range(10):
                    print('nop')
            self.execute(self.proc.wait)

        def test_proc_info(self):
            if False:
                return 10

            def call():
                if False:
                    return 10
                try:
                    return cext.proc_info(self.proc.pid)
                except ProcessLookupError:
                    pass
            self.execute(call)

@unittest.skipIf(not WINDOWS, 'WINDOWS only')
class TestProcessDualImplementation(TestMemoryLeak):

    def test_cmdline_peb_true(self):
        if False:
            for i in range(10):
                print('nop')
        self.execute(lambda : cext.proc_cmdline(os.getpid(), use_peb=True))

    def test_cmdline_peb_false(self):
        if False:
            while True:
                i = 10
        self.execute(lambda : cext.proc_cmdline(os.getpid(), use_peb=False))

class TestModuleFunctionsLeaks(TestMemoryLeak):
    """Test leaks of psutil module functions."""

    def test_coverage(self):
        if False:
            print('Hello World!')
        ns = system_namespace()
        ns.test_class_coverage(self, ns.all)

    @fewtimes_if_linux()
    def test_cpu_count(self):
        if False:
            print('Hello World!')
        self.execute(lambda : psutil.cpu_count(logical=True))

    @fewtimes_if_linux()
    def test_cpu_count_cores(self):
        if False:
            print('Hello World!')
        self.execute(lambda : psutil.cpu_count(logical=False))

    @fewtimes_if_linux()
    def test_cpu_times(self):
        if False:
            while True:
                i = 10
        self.execute(psutil.cpu_times)

    @fewtimes_if_linux()
    def test_per_cpu_times(self):
        if False:
            for i in range(10):
                print('nop')
        self.execute(lambda : psutil.cpu_times(percpu=True))

    @fewtimes_if_linux()
    def test_cpu_stats(self):
        if False:
            print('Hello World!')
        self.execute(psutil.cpu_stats)

    @fewtimes_if_linux()
    @unittest.skipIf(MACOS and platform.machine() == 'arm64', 'skipped due to #1892')
    @unittest.skipIf(not HAS_CPU_FREQ, 'not supported')
    def test_cpu_freq(self):
        if False:
            return 10
        self.execute(psutil.cpu_freq)

    @unittest.skipIf(not WINDOWS, 'WINDOWS only')
    def test_getloadavg(self):
        if False:
            print('Hello World!')
        psutil.getloadavg()
        self.execute(psutil.getloadavg)

    def test_virtual_memory(self):
        if False:
            print('Hello World!')
        self.execute(psutil.virtual_memory)

    @unittest.skipIf(SUNOS, 'worthless on SUNOS (uses a subprocess)')
    def test_swap_memory(self):
        if False:
            return 10
        self.execute(psutil.swap_memory)

    def test_pid_exists(self):
        if False:
            return 10
        times = FEW_TIMES if POSIX else self.times
        self.execute(lambda : psutil.pid_exists(os.getpid()), times=times)

    def test_disk_usage(self):
        if False:
            while True:
                i = 10
        times = FEW_TIMES if POSIX else self.times
        self.execute(lambda : psutil.disk_usage('.'), times=times)

    def test_disk_partitions(self):
        if False:
            for i in range(10):
                print('nop')
        self.execute(psutil.disk_partitions)

    @unittest.skipIf(LINUX and (not os.path.exists('/proc/diskstats')), '/proc/diskstats not available on this Linux version')
    @fewtimes_if_linux()
    def test_disk_io_counters(self):
        if False:
            i = 10
            return i + 15
        self.execute(lambda : psutil.disk_io_counters(nowrap=False))

    @fewtimes_if_linux()
    def test_pids(self):
        if False:
            i = 10
            return i + 15
        self.execute(psutil.pids)

    @fewtimes_if_linux()
    @unittest.skipIf(not HAS_NET_IO_COUNTERS, 'not supported')
    def test_net_io_counters(self):
        if False:
            return 10
        self.execute(lambda : psutil.net_io_counters(nowrap=False))

    @fewtimes_if_linux()
    @unittest.skipIf(MACOS and os.getuid() != 0, 'need root access')
    def test_net_connections(self):
        if False:
            print('Hello World!')
        psutil.net_connections(kind='all')
        with create_sockets():
            self.execute(lambda : psutil.net_connections(kind='all'))

    def test_net_if_addrs(self):
        if False:
            for i in range(10):
                print('nop')
        tolerance = 80 * 1024 if WINDOWS else self.tolerance
        self.execute(psutil.net_if_addrs, tolerance=tolerance)

    def test_net_if_stats(self):
        if False:
            for i in range(10):
                print('nop')
        self.execute(psutil.net_if_stats)

    @fewtimes_if_linux()
    @unittest.skipIf(not HAS_SENSORS_BATTERY, 'not supported')
    def test_sensors_battery(self):
        if False:
            i = 10
            return i + 15
        self.execute(psutil.sensors_battery)

    @fewtimes_if_linux()
    @unittest.skipIf(not HAS_SENSORS_TEMPERATURES, 'not supported')
    def test_sensors_temperatures(self):
        if False:
            while True:
                i = 10
        self.execute(psutil.sensors_temperatures)

    @fewtimes_if_linux()
    @unittest.skipIf(not HAS_SENSORS_FANS, 'not supported')
    def test_sensors_fans(self):
        if False:
            for i in range(10):
                print('nop')
        self.execute(psutil.sensors_fans)

    @fewtimes_if_linux()
    def test_boot_time(self):
        if False:
            while True:
                i = 10
        self.execute(psutil.boot_time)

    def test_users(self):
        if False:
            i = 10
            return i + 15
        self.execute(psutil.users)

    def test_set_debug(self):
        if False:
            while True:
                i = 10
        self.execute(lambda : psutil._set_debug(False))
    if WINDOWS:

        def test_win_service_iter(self):
            if False:
                i = 10
                return i + 15
            self.execute(cext.winservice_enumerate)

        def test_win_service_get(self):
            if False:
                print('Hello World!')
            pass

        def test_win_service_get_config(self):
            if False:
                return 10
            name = next(psutil.win_service_iter()).name()
            self.execute(lambda : cext.winservice_query_config(name))

        def test_win_service_get_status(self):
            if False:
                return 10
            name = next(psutil.win_service_iter()).name()
            self.execute(lambda : cext.winservice_query_status(name))

        def test_win_service_get_description(self):
            if False:
                while True:
                    i = 10
            name = next(psutil.win_service_iter()).name()
            self.execute(lambda : cext.winservice_query_descr(name))
if __name__ == '__main__':
    from psutil.tests.runner import run_from_name
    run_from_name(__file__)