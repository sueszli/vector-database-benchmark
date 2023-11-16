from multiprocessing import Process
import os

def run_proc(name):
    if False:
        while True:
            i = 10
    print('Run child process %s (%s)...' % (name, os.getpid()))
if __name__ == '__main__':
    print('Parent process %s.' % os.getpid())
    p = Process(target=run_proc, args=('test',))
    print('Child process will start.')
    p.start()
    p.join()
    print('Child process end.')