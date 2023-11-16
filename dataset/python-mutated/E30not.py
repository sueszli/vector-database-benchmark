class X:
    pass

def foo():
    if False:
        while True:
            i = 10
    pass

class X:
    pass

def foo():
    if False:
        i = 10
        return i + 15
    pass

class X:

    def a():
        if False:
            return 10
        pass

    def b():
        if False:
            while True:
                i = 10
        pass

    def c():
        if False:
            i = 10
            return i + 15
        pass

@some_decorator
class Y:

    def a():
        if False:
            for i in range(10):
                print('nop')
        pass

    def b():
        if False:
            while True:
                i = 10
        pass

    @property
    def c():
        if False:
            while True:
                i = 10
        pass
try:
    from nonexistent import Bar
except ImportError:

    class Bar(object):
        """This is a Bar replacement"""

def with_feature(f):
    if False:
        return 10
    'Some decorator'
    wrapper = f
    if has_this_feature(f):

        def wrapper(*args):
            if False:
                while True:
                    i = 10
            call_feature(args[0])
            return f(*args)
    return wrapper
try:
    next
except NameError:

    def next(iterator, default):
        if False:
            for i in range(10):
                print('nop')
        for item in iterator:
            return item
        return default

def a():
    if False:
        i = 10
        return i + 15
    pass

class Foo:
    """Class Foo"""

    def b():
        if False:
            while True:
                i = 10
        pass

def c():
    if False:
        print('Hello World!')
    pass

def d():
    if False:
        for i in range(10):
            print('nop')
    pass

def e():
    if False:
        print('Hello World!')
    pass

def a():
    if False:
        for i in range(10):
            print('nop')
    print
    print
    print

def b():
    if False:
        while True:
            i = 10
    pass