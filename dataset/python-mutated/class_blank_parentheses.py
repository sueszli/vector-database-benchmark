class SimpleClassWithBlankParentheses:
    pass

class ClassWithSpaceParentheses:
    first_test_data = 90
    second_test_data = 100

    def test_func(self):
        if False:
            while True:
                i = 10
        return None

class ClassWithEmptyFunc(object):

    def func_with_blank_parentheses():
        if False:
            return 10
        return 5

def public_func_with_blank_parentheses():
    if False:
        print('Hello World!')
    return None

def class_under_the_func_with_blank_parentheses():
    if False:
        print('Hello World!')

    class InsideFunc:
        pass

class NormalClass:

    def func_for_testing(self, first, second):
        if False:
            return 10
        sum = first + second
        return sum

class SimpleClassWithBlankParentheses:
    pass

class ClassWithSpaceParentheses:
    first_test_data = 90
    second_test_data = 100

    def test_func(self):
        if False:
            while True:
                i = 10
        return None

class ClassWithEmptyFunc(object):

    def func_with_blank_parentheses():
        if False:
            while True:
                i = 10
        return 5

def public_func_with_blank_parentheses():
    if False:
        print('Hello World!')
    return None

def class_under_the_func_with_blank_parentheses():
    if False:
        return 10

    class InsideFunc:
        pass

class NormalClass:

    def func_for_testing(self, first, second):
        if False:
            print('Hello World!')
        sum = first + second
        return sum