import bench

class Foo:
    num = 20000000

def test(num):
    if False:
        while True:
            i = 10
    i = 0
    while i < Foo.num:
        i += 1
bench.run(test)