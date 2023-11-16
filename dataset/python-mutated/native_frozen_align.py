import micropython

@micropython.native
def native_x(x):
    if False:
        i = 10
        return i + 15
    print(x + 1)

@micropython.native
def native_y(x):
    if False:
        i = 10
        return i + 15
    print(x + 1)

@micropython.native
def native_z(x):
    if False:
        return 10
    print(x + 1)