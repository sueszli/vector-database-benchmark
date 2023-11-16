import bench

def test(num):
    if False:
        print('Hello World!')
    for i in iter(range(num // 10000)):
        l = [0] * 1000
        l2 = bytes(l)
bench.run(test)