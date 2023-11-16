import _thread
import time
import micropython
import gc
try:
    micropython.schedule
except AttributeError:
    print('SKIP')
    raise SystemExit
gc.disable()
_NUM_TASKS = 10000
_TIMEOUT_MS = 10000
n = 0
t = None

def task(x):
    if False:
        i = 10
        return i + 15
    global n
    n += 1

def thread():
    if False:
        return 10
    while True:
        try:
            micropython.schedule(task, None)
        except RuntimeError:
            time.sleep_ms(10)
for i in range(8):
    _thread.start_new_thread(thread, ())
t = time.ticks_ms()
while n < _NUM_TASKS and time.ticks_diff(time.ticks_ms(), t) < _TIMEOUT_MS:
    pass
if n < _NUM_TASKS:
    print(n)
else:
    print('PASS')