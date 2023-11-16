def a():
    if False:
        i = 10
        return i + 15
    return

def __init__():
    if False:
        return 10
    return

class A:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        return

class B:

    def __init__(self):
        if False:
            return 10
        return 3

    def gen(self):
        if False:
            while True:
                i = 10
        return 5

class MyClass:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        return 1

class MyClass2:
    """dummy class"""

    def __init__(self):
        if False:
            i = 10
            return i + 15
        return

class MyClass3:
    """dummy class"""

    def __init__(self):
        if False:
            return 10
        return None

class MyClass5:
    """dummy class"""

    def __init__(self):
        if False:
            return 10
        self.callable = lambda : (yield None)