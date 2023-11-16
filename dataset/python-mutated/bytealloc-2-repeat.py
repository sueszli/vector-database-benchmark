import bench

def test(num):
    if False:
        i = 10
        return i + 15
    for i in iter(range(num // 1000)):
        b'\x00' * 10000
bench.run(test)