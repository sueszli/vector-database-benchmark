import multiprocessing
import os
import threading
import traceback

def t():
    if False:
        while True:
            i = 10
    try:
        with multiprocessing.Pool(1):
            pass
    except Exception:
        traceback.print_exc()
        os._exit(1)

def main():
    if False:
        i = 10
        return i + 15
    threads = []
    for i in range(20):
        threads.append(threading.Thread(target=t))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
if __name__ == '__main__':
    main()