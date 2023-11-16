import threading
import time
from queue import Queue
COUNT = 10000

def worker(in_q, out_q):
    if False:
        i = 10
        return i + 15
    while True:
        job = in_q.get()
        out_q.put(job())

def main():
    if False:
        for i in range(10):
            print('nop')
    in_q = Queue()
    out_q = Queue()
    t = threading.Thread(target=worker, args=(in_q, out_q))
    t.start()
    while True:
        start = time.monotonic()
        for _ in range(COUNT):
            in_q.put(lambda : None)
            out_q.get()
        end = time.monotonic()
        print(f'{(end - start) / COUNT * 1000000.0:.2f} µs/job')
main()