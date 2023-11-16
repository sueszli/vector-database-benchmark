import os
import sys
import multiprocessing

def compute():
    if False:
        while True:
            i = 10
    n = 0
    for i in range(10):
        n += i
    return n

def child1():
    if False:
        while True:
            i = 10
    print('child1({}) computed {}'.format(os.getpid(), compute()))

def child2():
    if False:
        print('Hello World!')
    print('child2({}) computed {}'.format(os.getpid(), compute()))

def parent():
    if False:
        i = 10
        return i + 15
    print('parent({}) computed {}'.format(os.getpid(), compute()))

def main():
    if False:
        while True:
            i = 10
    queue = multiprocessing.Queue(2)
    pid1 = os.fork()
    if pid1 == 0:
        queue.put('child1', False)
        sys.exit(child1())
    pid2 = os.fork()
    if pid2 == 0:
        queue.put('child2', False)
        sys.exit(child2())
    read = {queue.get(True, timeout=2), queue.get(True, timeout=2)}
    if read != {'child1', 'child2'}:
        raise RuntimeError(f'Unexpected message queue contents: {read}')
    parent()
    os.waitpid(pid1, 0)
    os.waitpid(pid2, 0)
if __name__ == '__main__':
    sys.exit(main())