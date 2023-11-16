import bench

def test(num):
    if False:
        for i in range(10):
            print('nop')
    for i in iter(range(num // 10000)):
        ba = bytearray(b'\x00' * 1000)
        ba2 = bytearray(map(lambda x: x + 1, ba))
bench.run(test)