import time
import multiprocessing

def target():
    if False:
        print('Hello World!')
    pass
if __name__ == '__main__':
    multiprocessing.freeze_support()
    child = multiprocessing.Process(target=target)
    child.start()
    time.sleep(10000)
    child.join()