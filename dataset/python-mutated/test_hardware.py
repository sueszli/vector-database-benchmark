from unittest.mock import patch
from golem import hardware
from golem.appconfig import MIN_CPU_CORES, MIN_DISK_SPACE, MIN_MEMORY_SIZE
from golem.testutils import TempDirFixture

@patch('golem.hardware.cpus', return_value=[1] * 7)
@patch('golem.hardware.memory', return_value=70000000.0)
@patch('golem.hardware.disk', return_value=7000000000.0)
class TestHardware(TempDirFixture):

    def test_caps(self, *_):
        if False:
            i = 10
            return i + 15
        hardware.initialize(self.tempdir)
        caps = hardware.caps()
        assert caps['cpu_cores'] == 7
        assert caps['memory'] == 70000000.0
        assert caps['disk'] == 7000000000.0

    def test_cpu_cores(self, *_):
        if False:
            for i in range(10):
                print('nop')
        assert hardware.cap_cpus(-1) == MIN_CPU_CORES
        assert hardware.cap_cpus(0) == MIN_CPU_CORES
        assert hardware.cap_cpus(1) == 1
        assert hardware.cap_cpus(7) == 7
        assert hardware.cap_cpus(8) == 7
        assert hardware.cap_cpus(1000000000.0) == 7

    def test_memory(self, *_):
        if False:
            while True:
                i = 10
        assert hardware.cap_memory(-1) == MIN_MEMORY_SIZE
        assert hardware.cap_memory(1000000.0) == MIN_MEMORY_SIZE
        assert hardware.cap_memory(2 ** 20) == 2 ** 20
        assert hardware.cap_memory(10000000.0) == 10000000.0
        assert hardware.cap_memory(70000000.0) == 70000000.0
        assert hardware.cap_memory(9000000000.0) == 70000000.0

    def test_disk(self, *_):
        if False:
            while True:
                i = 10
        hardware.initialize(self.tempdir)
        assert hardware.cap_disk(-1) == MIN_DISK_SPACE
        assert hardware.cap_disk(1000000.0) == MIN_DISK_SPACE
        assert hardware.cap_disk(2 ** 20) == 2 ** 20
        assert hardware.cap_disk(10000000.0) == 10000000.0
        assert hardware.cap_disk(7000000000.0) == 7000000000.0
        assert hardware.cap_disk(9e+19) == 7000000000.0