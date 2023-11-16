import multiprocessing as mp
import numba

def w():
    if False:
        i = 10
        return i + 15
    pass

def main():
    if False:
        return 10
    ps = [mp.Process(target=w) for _ in range(4)]
    [p.start() for p in ps]
    [p.join() for p in ps]
if __name__ == '__main__':
    p = mp.get_context('spawn').Process(target=main)
    p.start()
    p.join()