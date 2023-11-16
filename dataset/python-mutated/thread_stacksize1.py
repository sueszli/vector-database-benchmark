import sys
import _thread
if sys.implementation.name == 'micropython':
    sz = 2 * 1024
else:
    sz = 512 * 1024

def foo():
    if False:
        while True:
            i = 10
    pass

def thread_entry():
    if False:
        i = 10
        return i + 15
    foo()
    with lock:
        global n_finished
        n_finished += 1
_thread.stack_size()
print(_thread.stack_size())
print(_thread.stack_size(sz))
print(_thread.stack_size() == sz)
print(_thread.stack_size())
lock = _thread.allocate_lock()
n_thread = 2
n_finished = 0
_thread.stack_size(sz)
for i in range(n_thread):
    while True:
        try:
            _thread.start_new_thread(thread_entry, ())
            break
        except OSError:
            pass
_thread.stack_size()
while n_finished < n_thread:
    pass
print('done')