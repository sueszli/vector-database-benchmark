import _thread

class User:

    def __init__(self):
        if False:
            print('Hello World!')
        self.a = 'A'
        self.b = 'B'
        self.c = 'C'
user = User()

def th(n, lo, hi):
    if False:
        for i in range(10):
            print('nop')
    for repeat in range(n):
        for i in range(lo, hi):
            setattr(user, 'attr_%u' % i, repeat + i)
            assert getattr(user, 'attr_%u' % i) == repeat + i
    with lock:
        global n_finished
        n_finished += 1
lock = _thread.allocate_lock()
n_repeat = 30
n_range = 50
n_thread = 4
n_finished = 0
for i in range(n_thread):
    _thread.start_new_thread(th, (n_repeat, i * n_range, (i + 1) * n_range))
while n_finished < n_thread:
    pass
print(user.a, user.b, user.c)
for i in range(n_thread * n_range):
    assert getattr(user, 'attr_%u' % i) == n_repeat - 1 + i