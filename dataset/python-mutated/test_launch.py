from torch.testing._internal.common_utils import TestCase, run_tests, IS_LINUX
import shutil
import subprocess
import tempfile
import unittest

@unittest.skipIf(not IS_LINUX, 'Only works on linux')
class TestTorchrun(TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_dir = tempfile.mkdtemp(prefix=self.__class__.__name__)

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        shutil.rmtree(self._test_dir)

    def test_cpu_info(self):
        if False:
            i = 10
            return i + 15
        lscpu_info = '# The following is the parsable format, which can be fed to other\n# programs. Each different item in every column has an unique ID\n# starting from zero.\n# CPU,Core,Socket,Node\n0,0,0,0\n1,1,0,0\n2,2,0,0\n3,3,0,0\n4,4,1,1\n5,5,1,1\n6,6,1,1\n7,7,1,1\n8,0,0,0\n9,1,0,0\n10,2,0,0\n11,3,0,0\n12,4,1,1\n13,5,1,1\n14,6,1,1\n15,7,1,1\n'
        from torch.backends.xeon.run_cpu import _CPUinfo
        cpuinfo = _CPUinfo(lscpu_info)
        assert cpuinfo._physical_core_nums() == 8
        assert cpuinfo._logical_core_nums() == 16
        assert cpuinfo.get_node_physical_cores(0) == [0, 1, 2, 3]
        assert cpuinfo.get_node_physical_cores(1) == [4, 5, 6, 7]
        assert cpuinfo.get_node_logical_cores(0) == [0, 1, 2, 3, 8, 9, 10, 11]
        assert cpuinfo.get_node_logical_cores(1) == [4, 5, 6, 7, 12, 13, 14, 15]
        assert cpuinfo.get_all_physical_cores() == [0, 1, 2, 3, 4, 5, 6, 7]
        assert cpuinfo.get_all_logical_cores() == [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15]
        assert cpuinfo.numa_aware_check([0, 1, 2, 3]) == [0]
        assert cpuinfo.numa_aware_check([4, 5, 6, 7]) == [1]
        assert cpuinfo.numa_aware_check([2, 3, 4, 5]) == [0, 1]

    def test_multi_threads(self):
        if False:
            while True:
                i = 10
        num = 0
        with subprocess.Popen(f'python -m torch.backends.xeon.run_cpu --ninstances 4 --use-default-allocator             --disable-iomp --disable-numactl --disable-taskset --log-path {self._test_dir} --no-python pwd', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as p:
            for line in p.stdout.readlines():
                segs = str(line, 'utf-8').strip().split('-')
                if segs[-1].strip() == 'pwd':
                    num += 1
        assert num == 4, 'Failed to launch multiple instances for inference'
if __name__ == '__main__':
    run_tests()