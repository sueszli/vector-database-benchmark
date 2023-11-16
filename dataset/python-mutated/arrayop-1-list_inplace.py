import bench

def test(num):
    if False:
        i = 10
        return i + 15
    for i in iter(range(num // 10000)):
        arr = [0] * 1000
        for i in range(len(arr)):
            arr[i] += 1
bench.run(test)