"""AIX specific tests."""
import re
import unittest
import psutil
from psutil import AIX
from psutil.tests import PsutilTestCase
from psutil.tests import sh

@unittest.skipIf(not AIX, 'AIX only')
class AIXSpecificTestCase(PsutilTestCase):

    def test_virtual_memory(self):
        if False:
            while True:
                i = 10
        out = sh('/usr/bin/svmon -O unit=KB')
        re_pattern = 'memory\\s*'
        for field in 'size inuse free pin virtual available mmode'.split():
            re_pattern += '(?P<%s>\\S+)\\s+' % (field,)
        matchobj = re.search(re_pattern, out)
        self.assertIsNotNone(matchobj, 'svmon command returned unexpected output')
        KB = 1024
        total = int(matchobj.group('size')) * KB
        available = int(matchobj.group('available')) * KB
        used = int(matchobj.group('inuse')) * KB
        free = int(matchobj.group('free')) * KB
        psutil_result = psutil.virtual_memory()
        TOLERANCE_SYS_MEM = 2 * KB * KB
        self.assertEqual(psutil_result.total, total)
        self.assertAlmostEqual(psutil_result.used, used, delta=TOLERANCE_SYS_MEM)
        self.assertAlmostEqual(psutil_result.available, available, delta=TOLERANCE_SYS_MEM)
        self.assertAlmostEqual(psutil_result.free, free, delta=TOLERANCE_SYS_MEM)

    def test_swap_memory(self):
        if False:
            print('Hello World!')
        out = sh('/usr/sbin/lsps -a')
        matchobj = re.search('(?P<space>\\S+)\\s+(?P<vol>\\S+)\\s+(?P<vg>\\S+)\\s+(?P<size>\\d+)MB', out)
        self.assertIsNotNone(matchobj, 'lsps command returned unexpected output')
        total_mb = int(matchobj.group('size'))
        MB = 1024 ** 2
        psutil_result = psutil.swap_memory()
        self.assertEqual(int(psutil_result.total / MB), total_mb)

    def test_cpu_stats(self):
        if False:
            return 10
        out = sh('/usr/bin/mpstat -a')
        re_pattern = 'ALL\\s*'
        for field in 'min maj mpcs mpcr dev soft dec ph cs ics bound rq push S3pull S3grd S0rd S1rd S2rd S3rd S4rd S5rd sysc'.split():
            re_pattern += '(?P<%s>\\S+)\\s+' % (field,)
        matchobj = re.search(re_pattern, out)
        self.assertIsNotNone(matchobj, 'mpstat command returned unexpected output')
        CPU_STATS_TOLERANCE = 1000
        psutil_result = psutil.cpu_stats()
        self.assertAlmostEqual(psutil_result.ctx_switches, int(matchobj.group('cs')), delta=CPU_STATS_TOLERANCE)
        self.assertAlmostEqual(psutil_result.syscalls, int(matchobj.group('sysc')), delta=CPU_STATS_TOLERANCE)
        self.assertAlmostEqual(psutil_result.interrupts, int(matchobj.group('dev')), delta=CPU_STATS_TOLERANCE)
        self.assertAlmostEqual(psutil_result.soft_interrupts, int(matchobj.group('soft')), delta=CPU_STATS_TOLERANCE)

    def test_cpu_count_logical(self):
        if False:
            i = 10
            return i + 15
        out = sh('/usr/bin/mpstat -a')
        mpstat_lcpu = int(re.search('lcpu=(\\d+)', out).group(1))
        psutil_lcpu = psutil.cpu_count(logical=True)
        self.assertEqual(mpstat_lcpu, psutil_lcpu)

    def test_net_if_addrs_names(self):
        if False:
            while True:
                i = 10
        out = sh('/etc/ifconfig -l')
        ifconfig_names = set(out.split())
        psutil_names = set(psutil.net_if_addrs().keys())
        self.assertSetEqual(ifconfig_names, psutil_names)
if __name__ == '__main__':
    from psutil.tests.runner import run_from_name
    run_from_name(__file__)