import sys
import time
import multiprocessing

def test_function():
    if False:
        return 10
    time.sleep(1)
    print('In subprocess')

def main(start_method):
    if False:
        for i in range(10):
            print('nop')
    multiprocessing.set_start_method(start_method)
    print('In main')
    process = multiprocessing.Process(target=test_function)
    process.start()
    process.join()
    assert process.exitcode == 0, f'Process exited with non-success code {process.exitcode}!'
if __name__ == '__main__':
    multiprocessing.freeze_support()
    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} <start-method>')
        sys.exit(1)
    main(sys.argv[1])