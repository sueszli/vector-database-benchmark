import bench

class Foo:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._num = 20000000

    def num(self):
        if False:
            return 10
        return self._num

def test(num):
    if False:
        for i in range(10):
            print('nop')
    o = Foo()
    i = 0
    while i < o.num():
        i += 1
bench.run(test)