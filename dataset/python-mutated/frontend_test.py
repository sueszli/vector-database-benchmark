import pytest
import frontend

class FakeTime(object):
    """Fake implementations of GetUserCpuTime, GetUserCpuTime and BusyWait.
    Each call to BusyWait advances both the cpu and the wall clocks by fixed
    intervals (cpu_time_step and wall_time_step, respectively). This can be
    used to simulate arbitrary fraction of CPU time available to the process.
    """

    def __init__(self, cpu_time_step=1.0, wall_time_step=1.0):
        if False:
            i = 10
            return i + 15
        self.cpu_time = 0.0
        self.wall_time = 0.0
        self.cpu_time_step = cpu_time_step
        self.wall_time_step = wall_time_step

    def get_walltime(self):
        if False:
            print('Hello World!')
        return self.wall_time

    def get_user_cputime(self):
        if False:
            for i in range(10):
                print('nop')
        return self.cpu_time

    def busy_wait(self):
        if False:
            print('Hello World!')
        self.wall_time += self.wall_time_step
        self.cpu_time += self.cpu_time_step

@pytest.fixture
def faketime():
    if False:
        return 10
    return FakeTime()

@pytest.fixture
def cpuburner(faketime):
    if False:
        print('Hello World!')
    cpuburner = frontend.CpuBurner()
    cpuburner.get_user_cputime = faketime.get_user_cputime
    cpuburner.get_walltime = faketime.get_walltime
    cpuburner.busy_wait = faketime.busy_wait
    return cpuburner

def test_ok_response(faketime, cpuburner):
    if False:
        i = 10
        return i + 15
    faketime.cpu_time_step = 0.25
    (code, _) = cpuburner.handle_http_request()
    assert code == 200

def test_timeout(faketime, cpuburner):
    if False:
        for i in range(10):
            print('nop')
    faketime.cpu_time_step = 0.15
    (code, _) = cpuburner.handle_http_request()
    assert code == 500