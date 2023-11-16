import bench

def func(a, b, c):
    if False:
        return 10
    pass

def test(num):
    if False:
        i = 10
        return i + 15
    for i in iter(range(num)):
        func(c=i, b=i, a=i)
bench.run(test)