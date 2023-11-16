@micropython.viper
def f():
    if False:
        return 10
    return 123456789012345678901234567890
print(f())