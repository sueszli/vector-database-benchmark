import time
if hasattr(time, 'sleep_ms'):
    sleep_ms = time.sleep_ms
else:
    sleep_ms = lambda t: time.sleep(t / 1000)
import _thread

def thread_entry(n):
    if False:
        i = 10
        return i + 15
    pass
thread_num = 0
while thread_num < 500:
    try:
        _thread.start_new_thread(thread_entry, (thread_num,))
        thread_num += 1
    except (MemoryError, OSError) as er:
        sleep_ms(50)
sleep_ms(500)
print('done')