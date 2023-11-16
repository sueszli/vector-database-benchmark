import time
import sys
EPSILON = 1e-06
if sys.platform == 'win32' or sys.platform == 'cygwin':
    CLOCK_INACCURACY = 0.019
else:
    CLOCK_INACCURACY = 0.005

def test_clock_get_frame_time(clockobj):
    if False:
        i = 10
        return i + 15
    current_time = clockobj.get_frame_time()
    time.sleep(0.2)
    assert clockobj.get_frame_time() == current_time

def test_clock_jump_frame_time(clockobj):
    if False:
        i = 10
        return i + 15
    current_time = clockobj.get_frame_time()
    clockobj.tick()
    assert clockobj.get_frame_time() == current_time + clockobj.get_frame_time()

def test_clock_get_real_time(clockobj):
    if False:
        while True:
            i = 10
    current_time = clockobj.get_real_time()
    time.sleep(0.4)
    assert clockobj.get_real_time() - current_time + EPSILON >= 0.4 - CLOCK_INACCURACY

def test_clock_get_long_time(clockobj):
    if False:
        return 10
    current_time = clockobj.get_long_time()
    time.sleep(0.4)
    assert clockobj.get_long_time() - current_time + EPSILON >= 0.4 - CLOCK_INACCURACY

def test_clock_get_dt(clockobj):
    if False:
        print('Hello World!')
    clockobj.tick()
    first_tick = clockobj.get_frame_time()
    clockobj.tick()
    second_tick = clockobj.get_frame_time()
    assert clockobj.get_dt() == second_tick - first_tick

def test_clock_reset(clockobj):
    if False:
        print('Hello World!')
    clockobj.reset()
    assert clockobj.get_dt() == 0
    assert clockobj.get_frame_time() == 0
    assert clockobj.get_real_time() - EPSILON <= CLOCK_INACCURACY