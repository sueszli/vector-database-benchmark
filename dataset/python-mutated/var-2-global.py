import bench
ITERS = 20000000

def test(num):
    if False:
        print('Hello World!')
    i = 0
    while i < ITERS:
        i += 1
bench.run(test)