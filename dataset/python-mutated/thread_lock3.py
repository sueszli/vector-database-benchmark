import _thread
lock = _thread.allocate_lock()
n_thread = 10
n_finished = 0

def thread_entry(idx):
    if False:
        i = 10
        return i + 15
    global n_finished
    while True:
        with lock:
            if n_finished == idx:
                break
    print('my turn:', idx)
    with lock:
        n_finished += 1
for i in range(n_thread):
    _thread.start_new_thread(thread_entry, (i,))
while n_finished < n_thread:
    pass