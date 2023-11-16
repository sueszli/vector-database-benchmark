from multiprocessing import Pool

def f(n):
    if False:
        i = 10
        return i + 15
    return n * n
if __name__ == '__main__':
    p = Pool(processes=3)
    result = p.map(f, [1, 2, 3, 4, 5])
    for n in result:
        print(n)