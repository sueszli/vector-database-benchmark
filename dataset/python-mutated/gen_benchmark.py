from timeit import Timer
from tornado import gen
from tornado.options import options, define, parse_command_line
define('num', default=10000, help='number of iterations')

@gen.engine
def e2(callback):
    if False:
        i = 10
        return i + 15
    callback()

@gen.engine
def e1():
    if False:
        for i in range(10):
            print('nop')
    for i in range(10):
        yield gen.Task(e2)

@gen.coroutine
def c2():
    if False:
        print('Hello World!')
    pass

@gen.coroutine
def c1():
    if False:
        print('Hello World!')
    for i in range(10):
        yield c2()

def main():
    if False:
        i = 10
        return i + 15
    parse_command_line()
    t = Timer(e1)
    results = t.timeit(options.num) / options.num
    print('engine: %0.3f ms per iteration' % (results * 1000))
    t = Timer(c1)
    results = t.timeit(options.num) / options.num
    print('coroutine: %0.3f ms per iteration' % (results * 1000))
if __name__ == '__main__':
    main()