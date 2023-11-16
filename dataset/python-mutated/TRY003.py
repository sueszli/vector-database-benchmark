class CustomException(Exception):
    pass

def func():
    if False:
        i = 10
        return i + 15
    a = 1
    if a == 1:
        raise CustomException('Long message')
    elif a == 2:
        raise CustomException('Short')
    elif a == 3:
        raise CustomException('its_code_not_message')

def ignore():
    if False:
        while True:
            i = 10
    try:
        a = 1
    except Exception as ex:
        raise ex

class BadArgCantBeEven(Exception):
    pass

class GoodArgCantBeEven(Exception):

    def __init__(self, arg):
        if False:
            return 10
        super().__init__(f"The argument '{arg}' should be even")

def bad(a):
    if False:
        return 10
    if a % 2 == 0:
        raise BadArgCantBeEven(f"The argument '{a}' should be even")

def another_bad(a):
    if False:
        print('Hello World!')
    if a % 2 == 0:
        raise BadArgCantBeEven(f'The argument {a} should not be odd.')

def and_another_bad(a):
    if False:
        i = 10
        return i + 15
    if a % 2 == 0:
        raise BadArgCantBeEven('The argument `a` should not be odd.')

def good(a: int):
    if False:
        print('Hello World!')
    if a % 2 == 0:
        raise GoodArgCantBeEven(a)

def another_good(a):
    if False:
        i = 10
        return i + 15
    if a % 2 == 0:
        raise GoodArgCantBeEven(a)

def another_good():
    if False:
        return 10
    raise NotImplementedError('This is acceptable too')