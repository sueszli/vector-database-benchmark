import time
import _thread

def check(s, val):
    if False:
        while True:
            i = 10
    assert type(s) == str
    assert int(s) == val

def th(base, n):
    if False:
        return 10
    for i in range(n):
        exec("check('%u', %u)" % (base + i, base + i))
    with lock:
        global n_finished
        n_finished += 1
lock = _thread.allocate_lock()
n_thread = 4
n_finished = 0
n_qstr_per_thread = 100
for i in range(n_thread):
    _thread.start_new_thread(th, (i * n_qstr_per_thread, n_qstr_per_thread))
while n_finished < n_thread:
    time.sleep(1)
print('pass')