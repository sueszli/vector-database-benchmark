def scope0():
    if False:
        while True:
            i = 10
    any(((var0 := i) for i in range(2)))
    return var0
print('scope0')
print(scope0())
print(globals().get('var0', None))

def scope1():
    if False:
        print('Hello World!')
    var1 = 0
    dummy1 = 1
    dummy2 = 1
    print([(var1 := i) for i in [0, 1] if i > var1])
    print(var1)
print('scope1')
scope1()
print(globals().get('var1', None))

def scope2():
    if False:
        i = 10
        return i + 15
    global var2
    print([(var2 := i) for i in range(2)])
    print(globals().get('var2', None))
print('scope2')
scope2()
print(globals().get('var2', None))

def scope3():
    if False:
        print('Hello World!')
    global var3

    def inner3():
        if False:
            for i in range(10):
                print('nop')
        print([(var3 := i) for i in range(2)])
    inner3()
    print(globals().get('var3', None))
print('scope3')
scope3()
print(globals().get('var3', None))

def scope4():
    if False:
        while True:
            i = 10
    global var4

    def inner4():
        if False:
            i = 10
            return i + 15
        global var4
        print([(var4 := i) for i in range(2)])
    inner4()
    print(var4)
print('scope4')
scope4()
print(globals().get('var4', None))

def scope5():
    if False:
        while True:
            i = 10

    def inner5():
        if False:
            return 10
        nonlocal var5
        print([(var5 := i) for i in range(2)])
    inner5()
    print(var5)
    var5 = 0
print('scope5')
scope5()
print(globals().get('var5', None))