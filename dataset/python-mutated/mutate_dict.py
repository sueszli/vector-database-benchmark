import _thread
di = {'a': 'A', 'b': 'B', 'c': 'C', 'd': 'D'}

def th(n, lo, hi):
    if False:
        while True:
            i = 10
    for repeat in range(n):
        for i in range(lo, hi):
            di[i] = repeat + i
            assert di[i] == repeat + i
            del di[i]
            assert i not in di
            di[i] = repeat + i
            assert di[i] == repeat + i
            assert di.pop(i) == repeat + i
    with lock:
        global n_finished
        n_finished += 1
lock = _thread.allocate_lock()
n_thread = 4
n_finished = 0
for i in range(n_thread):
    _thread.start_new_thread(th, (30, i * 300, (i + 1) * 300))
while n_finished < n_thread:
    pass
print(sorted(di.items()))