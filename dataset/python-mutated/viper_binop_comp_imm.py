@micropython.viper
def f(a: int):
    if False:
        return 10
    print(a == -1, a == -255, a == -256, a == -257)
f(-1)
f(-255)
f(-256)
f(-257)