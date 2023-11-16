def f(x):
    if False:
        return 10
    return x * x
print(list(map(f, [1, 2, 3, 4, 5, 6, 7, 8, 9])))