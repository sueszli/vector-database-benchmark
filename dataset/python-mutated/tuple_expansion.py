def foo():
    if False:
        print('Hello World!')
    a = (1,)
    return (2, *a)
print()