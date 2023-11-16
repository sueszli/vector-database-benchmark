import bench

def test(num):
    if False:
        for i in range(10):
            print('nop')
    i = 0
    while i < 20000000:
        i += 1
bench.run(test)