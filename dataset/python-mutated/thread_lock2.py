import time
import _thread
lock = _thread.allocate_lock()

def thread_entry():
    if False:
        while True:
            i = 10
    lock.acquire()
    print('have it')
    lock.release()
for i in range(4):
    _thread.start_new_thread(thread_entry, ())
time.sleep(1)
print('done')