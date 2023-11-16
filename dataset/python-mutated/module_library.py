ROBOT_LIBRARY_SCOPE = 'Test Suite'
__version__ = 'test'

def passing():
    if False:
        while True:
            i = 10
    pass

def failing():
    if False:
        return 10
    raise AssertionError('This is a failing keyword from module library')

def logging():
    if False:
        for i in range(10):
            print('nop')
    print('Hello from module library')
    print('*WARN* WARNING!')

def returning():
    if False:
        print('Hello World!')
    return 'Hello from module library'

def argument(arg):
    if False:
        return 10
    assert arg == 'Hello', "Expected 'Hello', got '%s'" % arg

def many_arguments(arg1, arg2, arg3):
    if False:
        while True:
            i = 10
    assert arg1 == arg2 == arg3, 'All arguments should have been equal, got: %s, %s and %s' % (arg1, arg2, arg3)

def default_arguments(arg1, arg2='Hi', arg3='Hello'):
    if False:
        while True:
            i = 10
    many_arguments(arg1, arg2, arg3)

def variable_arguments(*args):
    if False:
        i = 10
        return i + 15
    return sum([int(arg) for arg in args])
attribute = 'This is not a keyword!'

class NotLibrary:

    def two_arguments(self, arg1, arg2):
        if False:
            i = 10
            return i + 15
        msg = "Arguments should have been unequal, both were '%s'" % arg1
        assert arg1 != arg2, msg

    def not_keyword(self):
        if False:
            return 10
        pass
notlib = NotLibrary()
two_arguments_from_class = notlib.two_arguments
lambda_keyword = lambda arg: int(arg) + 1
lambda_keyword_with_two_args = lambda x, y: int(x) / int(y)

def _not_keyword():
    if False:
        return 10
    pass

def module_library():
    if False:
        print('Hello World!')
    return 'It should be OK to have an attribute with same name as the module'