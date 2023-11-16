import bench

def func(a):
    if False:
        while True:
            i = 10
    pass

def test(num):
    if False:
        for i in range(10):
            print('nop')
    for i in iter(range(num)):
        func(a=i)
bench.run(test)