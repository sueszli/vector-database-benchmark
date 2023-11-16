def function_1():
    if False:
        i = 10
        return i + 15
    function_3(1, 2)

def function_2():
    if False:
        while True:
            i = 10
    function_1()

def function_3(dummy, dummy2):
    if False:
        i = 10
        return i + 15
    pass

def function_4(**dummy):
    if False:
        i = 10
        return i + 15
    return 1
    return 2

def function_5(dummy, dummy2, **dummy3):
    if False:
        return 10
    if False:
        return 7
    return 8

def start():
    if False:
        print('Hello World!')
    function_1()
    function_2()
    function_3(1, 2)
    function_4(test=42)
    function_5(*(1, 2), **{'test': 42})
start()