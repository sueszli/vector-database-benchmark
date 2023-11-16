import _thread
li = list()

def th(n, lo, hi):
    if False:
        return 10
    for repeat in range(n):
        for i in range(lo, hi):
            li.append(i)
            assert li.count(i) == repeat + 1
            li.extend([i, i])
            assert li.count(i) == repeat + 3
            li.remove(i)
            assert li.count(i) == repeat + 2
            li.remove(i)
            assert li.count(i) == repeat + 1
    with lock:
        global n_finished
        n_finished += 1
lock = _thread.allocate_lock()
n_thread = 4
n_finished = 0
for i in range(n_thread):
    _thread.start_new_thread(th, (4, i * 60, (i + 1) * 60))
while n_finished < n_thread:
    pass
li.sort()
print(li)