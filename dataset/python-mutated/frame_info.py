import inspect
from timeit import Timer
from pyinstrument.low_level.stat_profile import get_frame_info
frame = inspect.currentframe()
assert frame

def test_func():
    if False:
        for i in range(10):
            print('nop')
    get_frame_info(frame)
t = Timer(stmt=test_func)
test_func_timings = t.repeat(number=400000)
print('min time', min(test_func_timings))
print('max time', max(test_func_timings))
print('average time', sum(test_func_timings) / len(test_func_timings))