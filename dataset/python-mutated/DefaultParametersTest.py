""" Tests to cover default parameter behaviors.

"""
from __future__ import print_function
module_level = 1

def defaultValueTest1(no_default, some_default_constant=1):
    if False:
        i = 10
        return i + 15
    return some_default_constant

def defaultValueTest2(no_default, some_default_computed=module_level * 2):
    if False:
        i = 10
        return i + 15
    local_var = no_default
    return (local_var, some_default_computed)

def defaultValueTest3(no_default, func_defaulted=defaultValueTest1(module_level)):
    if False:
        for i in range(10):
            print('nop')
    return [func_defaulted for _i in range(8)]

def defaultValueTest4(no_default, lambda_defaulted=lambda x: x ** 2):
    if False:
        while True:
            i = 10
    c = 1
    d = 1
    return (i + c + d for i in range(8))

def defaultValueTest5(no_default, tuple_defaulted=(1, 2, 3)):
    if False:
        i = 10
        return i + 15
    return tuple_defaulted

def defaultValueTest6(no_default, list_defaulted=[1, 2, 3]):
    if False:
        print('Hello World!')
    list_defaulted.append(5)
    return list_defaulted
print(defaultValueTest1('ignored'))
module_level = 7
print(defaultValueTest2('also ignored'))
print(defaultValueTest3('no-no not again'))
print(list(defaultValueTest4('unused')))
print(defaultValueTest5('unused'))
print(defaultValueTest6('unused'), end='')
print(defaultValueTest6('unused'))
print(defaultValueTest6.__defaults__)
defaultValueTest6.func_defaults = ([1, 2, 3],)
print(defaultValueTest6.__defaults__)
print(defaultValueTest6(1))