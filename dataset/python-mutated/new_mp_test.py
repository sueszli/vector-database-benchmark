import multiprocessing
from time import sleep, perf_counter

class Integer(object):

    def __init__(self, x):
        if False:
            for i in range(10):
                print('nop')
        self.x = x

def largest_prime_factor(n):
    if False:
        i = 10
        return i + 15
    for i in range(10):
        x = [Integer(i * i) for i in range(80000)]
        a = x[50]
        print('\x1b[91mprogress ', n, i, a.x, '\x1b[0m')
    print('Done')
range_obj = range(4)
if __name__ == '__main__':
    t0 = perf_counter()
    handles = [multiprocessing.Process(target=largest_prime_factor, args=(i,)) for i in range_obj]
    for handle in handles:
        handle.start()
    for handle in handles:
        handle.join()
    dt = perf_counter() - t0
    print(f'Total time: {dt}')