import time
import _thread

def last(l):
    if False:
        while True:
            i = 10
    return l[-1]

def thread_entry(n):
    if False:
        print('Hello World!')
    data = bytearray((i for i in range(256)))
    lst = 8 * [0]
    sum = 0
    for i in range(n):
        sum += last(lst)
        lst = [0, 0, 0, 0, 0, 0, 0, i + 1]
    for (i, b) in enumerate(data):
        assert i == b
    with lock:
        print(sum, lst[-1])
        global n_finished
        n_finished += 1
lock = _thread.allocate_lock()
n_thread = 10
n_finished = 0
for i in range(n_thread):
    _thread.start_new_thread(thread_entry, (10000,))
while n_finished < n_thread:
    time.sleep(1)