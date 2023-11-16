import time
import multiprocessing

def target():
    if False:
        for i in range(10):
            print('nop')
    multiprocessing.freeze_support()
    time.sleep(1000)

def main():
    if False:
        return 10
    child1 = multiprocessing.Process(target=target)
    child1.start()
    child2 = multiprocessing.Process(target=target)
    child2.start()
    time.sleep(10000)
    child1.join()
    child2.join()
if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()