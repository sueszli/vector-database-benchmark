import time, threading
balance = 0
lock = threading.Lock()

def change_it(n):
    if False:
        return 10
    global balance
    balance = balance + n
    balance = balance - n

def run_thread(n):
    if False:
        print('Hello World!')
    for i in range(100000):
        lock.acquire()
        try:
            change_it(n)
        finally:
            lock.release()
t1 = threading.Thread(target=run_thread, args=(5,))
t2 = threading.Thread(target=run_thread, args=(8,))
t1.start()
t2.start()
t1.join()
t2.join()
print(balance)