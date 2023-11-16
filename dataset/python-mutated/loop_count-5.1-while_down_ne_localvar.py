import bench

def test(num):
    if False:
        print('Hello World!')
    zero = 0
    while num != zero:
        num -= 1
bench.run(test)