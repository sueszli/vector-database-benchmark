import socket
import time

def countdown(n):
    if False:
        return 10
    while n > 0:
        print('T-minus', n)
        n -= 1
        time.sleep(5)
from threading import Thread
t = Thread(target=countdown, args=(10,))
t.start()
print('Still running' if t.is_alive() else 'Completed')
t.join()
t = Thread(target=countdown, args=(10,), daemon=True)
t.start()

class CountdownTask:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._running = True

    def terminate(self):
        if False:
            return 10
        self._running = False

    def run(self, n):
        if False:
            print('Hello World!')
        while self._running and n > 0:
            print('T-minus', n)
            n -= 1
            time.sleep(5)
c = CountdownTask()
t = Thread(target=c.run, args=(10,))
t.start()
c.terminate()
t.join()

class IOTask:

    def terminate(self):
        if False:
            print('Hello World!')
        sock.settimeout(5)
        while self._running:
            try:
                data = sock.recv(8192)
                break
            except socket.timeout:
                continue
        return
from threading import Thread

class CountdownThread(Thread):

    def __init__(self, n):
        if False:
            return 10
        super().__init__()
        self.n = n

    def run(self):
        if False:
            return 10
        while self.n > 0:
            print('T-minus', self.n)
            self.n -= 1
            time.sleep(5)
c = CountdownThread(5)
c.start()
import multiprocessing
c = CountdownTask(5)
p = multiprocessing.Process(target=c.run)
p.start()