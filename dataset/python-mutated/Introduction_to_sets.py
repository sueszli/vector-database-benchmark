def average(a):
    if False:
        for i in range(10):
            print('nop')
    return sum(set(a)) / len(set(a))
if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    result = average(arr)
    print(result)