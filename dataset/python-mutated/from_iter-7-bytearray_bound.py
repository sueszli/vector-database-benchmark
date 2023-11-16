import bench

def test(num):
    if False:
        i = 10
        return i + 15
    for i in iter(range(num // 10000)):
        l = [0] * 1000
        l2 = bytearray(l)
bench.run(test)