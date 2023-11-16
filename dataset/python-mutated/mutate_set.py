import _thread
se = set([-1, -2, -3, -4])

def th(n, lo, hi):
    if False:
        while True:
            i = 10
    for repeat in range(n):
        for i in range(lo, hi):
            se.add(i)
            assert i in se
            se.remove(i)
            assert i not in se
    with lock:
        global n_finished
        n_finished += 1
lock = _thread.allocate_lock()
n_thread = 4
n_finished = 0
for i in range(n_thread):
    _thread.start_new_thread(th, (50, i * 500, (i + 1) * 500))
while n_finished < n_thread:
    pass
print(sorted(se))