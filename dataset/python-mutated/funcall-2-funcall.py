import bench

def f(x):
    if False:
        return 10
    return x + 1

def test(num):
    if False:
        i = 10
        return i + 15
    for i in iter(range(num)):
        a = f(i)
bench.run(test)