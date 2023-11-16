import bench

def test(num):
    if False:
        for i in range(10):
            print('nop')
    for i in iter(range(num)):
        pass
bench.run(test)