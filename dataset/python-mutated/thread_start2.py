import time
import _thread

def thread_entry(a0, a1, a2, a3):
    if False:
        print('Hello World!')
    print('thread', a0, a1, a2, a3)
_thread.start_new_thread(thread_entry, (10, 20), {'a2': 0, 'a3': 1})
time.sleep(1)
try:
    _thread.start_new_thread(thread_entry, (), ())
except TypeError:
    print('TypeError')
print('done')