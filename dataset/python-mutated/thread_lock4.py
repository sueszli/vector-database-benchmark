import time
import _thread

def fac(n):
    if False:
        while True:
            i = 10
    x = 1
    for i in range(1, n + 1):
        x *= i
    return x

def thread_entry():
    if False:
        i = 10
        return i + 15
    while True:
        with jobs_lock:
            try:
                (f, arg) = jobs.pop(0)
            except IndexError:
                return
        ans = f(arg)
        with output_lock:
            output.append((arg, ans))
jobs = [(fac, i) for i in range(20, 80)]
jobs_lock = _thread.allocate_lock()
n_jobs = len(jobs)
output = []
output_lock = _thread.allocate_lock()
for i in range(4):
    _thread.start_new_thread(thread_entry, ())
while True:
    with jobs_lock:
        if len(output) == n_jobs:
            break
    time.sleep(1)
output.sort(key=lambda x: x[0])
for (arg, ans) in output:
    print(arg, ans)