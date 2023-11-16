import time
import _thread

def thread_entry():
    if False:
        while True:
            i = 10
    raise ValueError
_thread.start_new_thread(thread_entry, ())
time.sleep(1)
print('done')