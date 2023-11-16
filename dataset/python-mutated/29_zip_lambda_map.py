a = [1, 2, 3]
b = [4, 5, 6]
list(zip(a, b))
list(zip(a, a, b))
for (i, j) in zip(a, b):
    print(i, j)

def f1(x, y):
    if False:
        print('Hello World!')
    return x + y
f2 = lambda x, y: x + y
print(f1(1, 2))
print(f2(1, 2))
print(list(map(f1, [1], [2])))
print(list(map(f2, [2, 3], [4, 5])))