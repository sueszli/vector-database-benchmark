import bench

def test(num):
    if False:
        return 10
    for i in iter(range(num // 10000)):
        arr = bytearray(b'\x00' * 1000)
        for i in range(len(arr)):
            arr[i] += 1
bench.run(test)