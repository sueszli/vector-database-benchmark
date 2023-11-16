import bench

def test(num):
    if False:
        i = 10
        return i + 15
    for i in iter(range(num // 10000)):
        arr = bytearray(b'\x00' * 1000)
        arr2 = bytearray(map(lambda x: x + 1, arr))
bench.run(test)