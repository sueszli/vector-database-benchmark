import bench

def test(num):
    if False:
        i = 10
        return i + 15
    for i in iter(range(num)):
        a = i + 1
bench.run(test)