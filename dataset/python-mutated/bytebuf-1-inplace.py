import bench

def test(num):
    if False:
        print('Hello World!')
    for i in iter(range(num // 10000)):
        ba = bytearray(b'\x00' * 1000)
        for i in range(len(ba)):
            ba[i] += 1
bench.run(test)