def execfile(fn):
    if False:
        return 10
    with open(fn) as f:
        exec(f.read())

def simple_test():
    if False:
        while True:
            i = 10
    execfile('examples/simple_test.py')

def mmap_test():
    if False:
        while True:
            i = 10
    execfile('examples/mmap_test.py')

def precision_test():
    if False:
        while True:
            i = 10
    execfile('examples/precision_test.py')