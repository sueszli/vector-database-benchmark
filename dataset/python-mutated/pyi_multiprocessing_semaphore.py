import sys
import multiprocessing

def test_function(semaphore):
    if False:
        for i in range(10):
            print('nop')
    semaphore.acquire()
    print('In subprocess')
    semaphore.release()

def main(start_method):
    if False:
        while True:
            i = 10
    multiprocessing.set_start_method(start_method)
    print('In main')
    sys.stdout.flush()
    semaphore = multiprocessing.Semaphore()
    semaphore.acquire()
    process = multiprocessing.Process(target=test_function, args=[semaphore])
    process.start()
    semaphore.release()
    process.join()
if __name__ == '__main__':
    multiprocessing.freeze_support()
    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} <start-method>')
        sys.exit(1)
    main(sys.argv[1])