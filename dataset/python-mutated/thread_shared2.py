import _thread

def foo(lst, i):
    if False:
        print('Hello World!')
    lst[i] += 1

def thread_entry(n, lst, idx):
    if False:
        print('Hello World!')
    for i in range(n):
        foo(lst, idx)
    with lock:
        global n_finished
        n_finished += 1
lock = _thread.allocate_lock()
n_thread = 2
n_finished = 0
lst = [0, 0]
for i in range(n_thread):
    _thread.start_new_thread(thread_entry, ((i + 1) * 10, lst, i))
while n_finished < n_thread:
    pass
print(lst)