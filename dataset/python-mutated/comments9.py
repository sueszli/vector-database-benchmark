some = statement

def function():
    if False:
        i = 10
        return i + 15
    pass
some = statement

def function():
    if False:
        print('Hello World!')
    pass
some = statement

async def async_function():
    pass
some = statement

class MyClass:
    pass
some = statement

class MyClassWithComplexLeadingComments:
    pass

class ClassWithDocstring:
    """A docstring."""

class MyClassAfterAnotherClassWithDocstring:
    pass
some = statement

@deco1
@deco2(with_args=True)
@deco3
def decorated():
    if False:
        for i in range(10):
            print('nop')
    pass
some = statement

@deco1
@deco2(with_args=True)
@deco3
def decorated_with_split_leading_comments():
    if False:
        i = 10
        return i + 15
    pass
some = statement

@deco1
@deco2(with_args=True)
@deco3
def decorated_with_split_leading_comments():
    if False:
        print('Hello World!')
    pass

def main():
    if False:
        i = 10
        return i + 15
    if a:

        def inline():
            if False:
                i = 10
                return i + 15
            pass

        def another_inline():
            if False:
                return 10
            pass
    else:

        def inline_after_else():
            if False:
                i = 10
                return i + 15
            pass
if a:

    def top_level_quote_inline():
        if False:
            return 10
        pass

    def another_top_level_quote_inline_inline():
        if False:
            print('Hello World!')
        pass
else:

    def top_level_quote_inline_after_else():
        if False:
            return 10
        pass

class MyClass:

    def first_method(self):
        if False:
            for i in range(10):
                print('nop')
        pass

def foo():
    if False:
        return 10
    pass

@decorator1
@decorator2
def bar():
    if False:
        while True:
            i = 10
    pass

def foo():
    if False:
        i = 10
        return i + 15
    pass

@decorator1
def bar():
    if False:
        while True:
            i = 10
    pass
some = statement

def function():
    if False:
        for i in range(10):
            print('nop')
    pass
some = statement

def function():
    if False:
        for i in range(10):
            print('nop')
    pass
some = statement

async def async_function():
    pass
some = statement

class MyClass:
    pass
some = statement

class MyClassWithComplexLeadingComments:
    pass

class ClassWithDocstring:
    """A docstring."""

class MyClassAfterAnotherClassWithDocstring:
    pass
some = statement

@deco1
@deco2(with_args=True)
@deco3
def decorated():
    if False:
        print('Hello World!')
    pass
some = statement

@deco1
@deco2(with_args=True)
@deco3
def decorated_with_split_leading_comments():
    if False:
        while True:
            i = 10
    pass
some = statement

@deco1
@deco2(with_args=True)
@deco3
def decorated_with_split_leading_comments():
    if False:
        return 10
    pass

def main():
    if False:
        return 10
    if a:

        def inline():
            if False:
                i = 10
                return i + 15
            pass

        def another_inline():
            if False:
                i = 10
                return i + 15
            pass
    else:

        def inline_after_else():
            if False:
                while True:
                    i = 10
            pass
if a:

    def top_level_quote_inline():
        if False:
            i = 10
            return i + 15
        pass

    def another_top_level_quote_inline_inline():
        if False:
            return 10
        pass
else:

    def top_level_quote_inline_after_else():
        if False:
            while True:
                i = 10
        pass

class MyClass:

    def first_method(self):
        if False:
            while True:
                i = 10
        pass

def foo():
    if False:
        return 10
    pass

@decorator1
@decorator2
def bar():
    if False:
        return 10
    pass

def foo():
    if False:
        i = 10
        return i + 15
    pass

@decorator1
def bar():
    if False:
        print('Hello World!')
    pass