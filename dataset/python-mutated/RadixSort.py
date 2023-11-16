def radix_sort_nums(L):
    if False:
        print('Hello World!')
    maxNum = L[0]
    for x in L:
        if maxNum < x:
            maxNum = x
    times = 0
    while maxNum > 0:
        maxNum = int(maxNum / 10)
        times += 1
    return times

def get_num_pos(num, pos):
    if False:
        for i in range(10):
            print('nop')
    return int(num / 10 ** (pos - 1)) % 10

def radix_sort(L):
    if False:
        print('Hello World!')
    count = 10 * [None]
    bucket = len(L) * [None]
    for pos in range(1, radix_sort_nums(L) + 1):
        for x in range(0, 10):
            count[x] = 0
        for x in range(0, len(L)):
            j = get_num_pos(int(L[x]), pos)
            count[j] += 1
        for x in range(1, 10):
            count[x] += count[x - 1]
        for x in range(len(L) - 1, -1, -1):
            j = get_num_pos(L[x], pos)
            bucket[count[j] - 1] = L[x]
            count[j] -= 1
        for x in range(0, len(L)):
            L[x] = bucket[x]