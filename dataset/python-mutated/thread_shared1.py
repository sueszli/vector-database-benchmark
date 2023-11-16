import _thread

def foo(i):
    if False:
        print('Hello World!')
    pass

def thread_entry(n, tup):
    if False:
        for i in range(10):
            print('nop')
    for i in tup:
        foo(i)
    with lock:
        global n_finished
        n_finished += 1
lock = _thread.allocate_lock()
n_thread = 2
n_finished = 0
tup = (1, 2, 3, 4)
for i in range(n_thread):
    _thread.start_new_thread(thread_entry, (100, tup))
while n_finished < n_thread:
    pass
print(tup)