def serviceLane(width, i, j):
    if False:
        for i in range(10):
            print('nop')
    return min(width[i:j + 1])
(n, t) = map(int, input().split())
width = list(map(int, input().split()))
for _ in range(t):
    (i, j) = map(int, input().split())
    print(serviceLane(width, i, j))