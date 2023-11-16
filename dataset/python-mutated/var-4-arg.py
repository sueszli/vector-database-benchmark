import bench

def test(num):
    if False:
        for i in range(10):
            print('nop')
    i = 0
    while i < num:
        i += 1
bench.run(lambda n: test(20000000))