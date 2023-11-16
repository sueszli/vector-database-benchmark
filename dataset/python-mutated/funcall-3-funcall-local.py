import bench

def f(x):
    if False:
        while True:
            i = 10
    return x + 1

def test(num):
    if False:
        i = 10
        return i + 15
    f_ = f
    for i in iter(range(num)):
        a = f_(i)
bench.run(test)