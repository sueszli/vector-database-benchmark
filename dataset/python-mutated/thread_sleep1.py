import time
if hasattr(time, 'sleep_ms'):
    sleep_ms = time.sleep_ms
else:
    sleep_ms = lambda t: time.sleep(t / 1000)
import _thread
lock = _thread.allocate_lock()
n_thread = 4
n_finished = 0

def thread_entry(t):
    if False:
        for i in range(10):
            print('nop')
    global n_finished
    sleep_ms(t)
    sleep_ms(2 * t)
    with lock:
        n_finished += 1
for i in range(n_thread):
    _thread.start_new_thread(thread_entry, (10 * i,))
while n_finished < n_thread:
    sleep_ms(100)
print('done', n_thread)