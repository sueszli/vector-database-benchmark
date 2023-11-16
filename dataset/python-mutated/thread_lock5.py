import _thread

def thread_entry():
    if False:
        i = 10
        return i + 15
    print('thread about to release lock')
    lock.release()
lock = _thread.allocate_lock()
lock.acquire()
_thread.start_new_thread(thread_entry, ())
lock.acquire()
print('main has lock')
lock.release()