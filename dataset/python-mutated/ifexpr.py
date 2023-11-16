print(1 if 0 else 2)
print(3 if 1 else 4)

def f(x):
    if False:
        return 10
    print('a' if x else 'b')
f([])
f([1])