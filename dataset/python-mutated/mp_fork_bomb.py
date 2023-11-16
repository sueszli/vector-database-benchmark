import multiprocessing, sys

def foo():
    if False:
        i = 10
        return i + 15
    print('123')
if len(sys.argv) > 1:
    multiprocessing.set_start_method(sys.argv[1])
else:
    multiprocessing.set_start_method('spawn')
p = multiprocessing.Process(target=foo)
p.start()
p.join()
sys.exit(p.exitcode)