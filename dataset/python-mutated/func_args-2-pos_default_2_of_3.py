import bench

def func(a, b=1, c=2):
    if False:
        print('Hello World!')
    pass

def test(num):
    if False:
        while True:
            i = 10
    for i in iter(range(num)):
        func(i)
bench.run(test)