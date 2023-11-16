import bench

def test(num):
    if False:
        return 10
    for i in iter(range(num // 10000)):
        ba = bytearray(b'\x00' * 1000)
        ba2 = b''.join(map(lambda x: bytes([x + 1]), ba))
bench.run(test)