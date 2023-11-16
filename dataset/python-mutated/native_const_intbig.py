@micropython.native
def f():
    if False:
        print('Hello World!')
    return 123456789012345678901234567890
print(f())