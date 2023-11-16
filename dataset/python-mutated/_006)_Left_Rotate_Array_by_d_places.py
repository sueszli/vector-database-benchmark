l = [10, 20, 30, 40, 50]
d = 3
print(l[d:] + l[0:d])
l = [10, 20, 30, 40, 50]
d = 3
for i in range(d):
    l.append(l.pop(0))
print(l)
l = [10, 20, 30, 40, 50]
d = 3
n = len(l)
for k in range(d):
    temp = l[0]
    for i in range(n - 1):
        l[i] = l[i + 1]
    l[n - 1] = temp
print(l)

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
        for i in range(10):
            print('nop')
    n = len(l)
    reverse(l, 0, d - 1)
    reverse(l, d, n - 1)
    reverse(l, 0, n - 1)
l = [10, 20, 30, 40, 50]
d = 3
optimized(l, d)
print(l)