import bench

def test(num):
    if False:
        for i in range(10):
            print('nop')
    for i in iter(range(num // 20)):
        enumerate([1, 2], 1)
bench.run(test)