import multiprocessing
from multiprocessing import Pool

def f(x):
    if False:
        print('Hello World!')
    print('Start')
    return [i for i in range(1000000)]
if __name__ == '__main__':
    with Pool(5) as p:
        q = p.map(f, [1, 2, 3])
        print(len(q))