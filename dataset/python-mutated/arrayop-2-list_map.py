import bench

def test(num):
    if False:
        i = 10
        return i + 15
    for i in iter(range(num // 10000)):
        arr = [0] * 1000
        arr2 = list(map(lambda x: x + 1, arr))
bench.run(test)