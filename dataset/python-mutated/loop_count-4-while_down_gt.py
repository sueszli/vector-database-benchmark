import bench

def test(num):
    if False:
        print('Hello World!')
    while num > 0:
        num -= 1
bench.run(test)