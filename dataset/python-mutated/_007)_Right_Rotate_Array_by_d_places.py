def reverse(lst, l, r):
    if False:
        for i in range(10):
            print('nop')
    while l < r:
        (lst[l], lst[r]) = (lst[r], lst[l])
        l += 1
        r -= 1

def optimized(l, d):
    if False:
        print('Hello World!')
    n = len(l)
    reverse(l, 0, n - 1)
    reverse(l, 0, d - 1)
    reverse(l, d, n - 1)
l = [10, 20, 30, 40, 50]
d = 3
optimized(l, d)
print(l)