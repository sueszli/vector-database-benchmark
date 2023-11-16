import sys
import multiprocessing

def f(x):
    if False:
        return 10
    return x * x

def main(start_method):
    if False:
        i = 10
        return i + 15
    multiprocessing.set_start_method(start_method)
    with multiprocessing.Pool(processes=4) as pool:
        print('Evaluate f(10) asynchronously...')
        result = pool.apply_async(f, [10])
        assert result.get(timeout=1) == 100
        print('Evaluate f(0..9)...')
        assert pool.map(f, range(10)) == [x ** 2 for x in range(10)]
if __name__ == '__main__':
    multiprocessing.freeze_support()
    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} <start-method>')
        sys.exit(1)
    main(sys.argv[1])