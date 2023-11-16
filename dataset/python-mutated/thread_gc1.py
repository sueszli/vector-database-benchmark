import gc
import _thread

def thread_entry(n):
    if False:
        for i in range(10):
            print('nop')
    data = bytearray((i for i in range(256)))
    for i in range(n):
        for i in range(len(data)):
            data[i] = data[i]
        gc.collect()
    with lock:
        print(list(data) == list(range(256)))
        global n_finished
        n_finished += 1
lock = _thread.allocate_lock()
n_thread = 4
n_finished = 0
for i in range(n_thread):
    _thread.start_new_thread(thread_entry, (10,))
while n_finished < n_thread:
    pass