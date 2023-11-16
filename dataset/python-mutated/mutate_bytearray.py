import _thread
ba = bytearray()

def th(n, lo, hi):
    if False:
        i = 10
        return i + 15
    for repeat in range(n):
        for i in range(lo, hi):
            l = len(ba)
            ba.append(i)
            assert len(ba) >= l + 1
            l = len(ba)
            ba.extend(bytearray([i]))
            assert len(ba) >= l + 1
    with lock:
        global n_finished
        n_finished += 1
lock = _thread.allocate_lock()
n_thread = 4
n_finished = 0
n_repeat = 4
for i in range(n_thread):
    _thread.start_new_thread(th, (n_repeat, i * 256 // n_thread, (i + 1) * 256 // n_thread))
while n_finished < n_thread:
    pass
print(len(ba))
count = [0 for _ in range(256)]
for b in ba:
    count[b] += 1
print(count)