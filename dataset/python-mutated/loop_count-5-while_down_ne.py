import bench

def test(num):
    if False:
        for i in range(10):
            print('nop')
    while num != 0:
        num -= 1
bench.run(test)