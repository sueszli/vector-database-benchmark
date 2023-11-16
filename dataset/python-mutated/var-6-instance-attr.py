import bench

class Foo:

    def __init__(self):
        if False:
            return 10
        self.num = 20000000

def test(num):
    if False:
        i = 10
        return i + 15
    o = Foo()
    i = 0
    while i < o.num:
        i += 1
bench.run(test)