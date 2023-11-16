from robot.api.deco import keyword

def defined_twice():
    if False:
        print('Hello World!')
    1 / 0

@keyword('Defined twice')
def this_time_using_custom_name():
    if False:
        print('Hello World!')
    2 / 0

def defined_thrice():
    if False:
        for i in range(10):
            print('nop')
    1 / 0

def definedThrice():
    if False:
        return 10
    2 / 0

def Defined_Thrice():
    if False:
        for i in range(10):
            print('nop')
    3 / 0

@keyword('Embedded ${arguments} twice')
def embedded1(arg):
    if False:
        while True:
            i = 10
    1 / 0

@keyword('Embedded ${arguments match} TWICE')
def embedded2(arg):
    if False:
        return 10
    2 / 0