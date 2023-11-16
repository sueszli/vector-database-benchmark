CONSTANT = 1

def FUNC():
    if False:
        return 10
    pass

class CLASS:
    pass

def fix_constant(value):
    if False:
        while True:
            i = 10
    'All this is ok, but try not to use `global` ;)'
    global CONSTANT
    print(CONSTANT)
    CONSTANT = value

def global_with_import():
    if False:
        while True:
            i = 10
    'Should only warn for global-statement when using `Import` node'
    global sys
    import sys

def global_with_import_from():
    if False:
        for i in range(10):
            print('nop')
    'Should only warn for global-statement when using `ImportFrom` node'
    global namedtuple
    from collections import namedtuple

def global_del():
    if False:
        for i in range(10):
            print('nop')
    'Deleting the global name prevents `global-variable-not-assigned`'
    global CONSTANT
    print(CONSTANT)
    del CONSTANT

def global_operator_assign():
    if False:
        print('Hello World!')
    'Operator assigns should only throw a global statement error'
    global CONSTANT
    print(CONSTANT)
    CONSTANT += 1

def global_function_assign():
    if False:
        print('Hello World!')
    'Function assigns should only throw a global statement error'
    global CONSTANT

    def CONSTANT():
        if False:
            while True:
                i = 10
        pass
    CONSTANT()

def override_func():
    if False:
        for i in range(10):
            print('nop')
    'Overriding a function should only throw a global statement error'
    global FUNC

    def FUNC():
        if False:
            while True:
                i = 10
        pass
    FUNC()

def override_class():
    if False:
        return 10
    'Overriding a class should only throw a global statement error'
    global CLASS

    class CLASS:
        pass
    CLASS()

def multiple_assignment():
    if False:
        return 10
    'Should warn on every assignment.'
    global CONSTANT
    CONSTANT = 1
    CONSTANT = 2

def no_assignment():
    if False:
        print('Hello World!')
    "Shouldn't warn"
    global CONSTANT