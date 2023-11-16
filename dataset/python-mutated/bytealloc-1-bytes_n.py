import bench

def test(num):
    if False:
        print('Hello World!')
    for i in iter(range(num // 1000)):
        bytes(10000)
bench.run(test)