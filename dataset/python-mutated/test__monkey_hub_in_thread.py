from gevent.monkey import patch_all
patch_all(thread=False)
from threading import Thread
import time

def func():
    if False:
        print('Hello World!')
    time.sleep()

def main():
    if False:
        i = 10
        return i + 15
    threads = []
    for _ in range(3):
        th = Thread(target=func)
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
if __name__ == '__main__':
    main()