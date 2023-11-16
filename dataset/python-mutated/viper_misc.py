import micropython

@micropython.viper
def viper_int(x: int, y: int) -> int:
    if False:
        i = 10
        return i + 15
    return x + y + 3
print(viper_int(1, 2))

@micropython.viper
def viper_object(x: object, y: object) -> object:
    if False:
        return 10
    return x + y
print(viper_object(1, 2))

@micropython.viper
def viper_ret_none() -> int:
    if False:
        i = 10
        return i + 15
    return None
print(viper_ret_none())

@micropython.viper
def viper_ret_ellipsis() -> object:
    if False:
        print('Hello World!')
    return ...
print(viper_ret_ellipsis())

@micropython.viper
def viper_3args(a: int, b: int, c: int) -> int:
    if False:
        return 10
    return a + b + c
print(viper_3args(1, 2, 3))

@micropython.viper
def viper_4args(a: int, b: int, c: int, d: int) -> int:
    if False:
        return 10
    return a + b + c + d
print(viper_4args(1, 2, 3, 4))

@micropython.viper
def viper_local(x: int) -> int:
    if False:
        print('Hello World!')
    y = 4
    return x + y
print(viper_local(3))

@micropython.viper
def viper_no_annotation(x, y):
    if False:
        while True:
            i = 10
    return x * y
print(viper_no_annotation(4, 5))