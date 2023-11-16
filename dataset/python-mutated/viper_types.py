import micropython

@micropython.viper
def f1(x: bool):
    if False:
        print('Hello World!')
    print(x)
f1(0)
f1(1)
f1([])
f1([1])

@micropython.viper
def f2(x: bool) -> bool:
    if False:
        while True:
            i = 10
    return x
print(f2([]))
print(f2([1]))

@micropython.viper
def f3(x) -> bool:
    if False:
        for i in range(10):
            print('nop')
    return bool(x)
print(f3([]))
print(f3(-1))