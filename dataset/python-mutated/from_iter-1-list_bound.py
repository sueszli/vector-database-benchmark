import bench

def test(num):
    if False:
        return 10
    for i in iter(range(num // 10000)):
        l = [0] * 1000
        l2 = list(l)
bench.run(test)