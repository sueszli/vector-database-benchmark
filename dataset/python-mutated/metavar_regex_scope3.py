def a():
    if False:
        for i in range(10):
            print('nop')
    return 1

def b():
    if False:
        return 10
    return

def c():
    if False:
        while True:
            i = 10
    return None

class A:

    def a(self):
        if False:
            for i in range(10):
                print('nop')
        return 1

class B:

    def b(self):
        if False:
            while True:
                i = 10
        return

class C:

    def c(self):
        if False:
            i = 10
            return i + 15
        return None

class D:

    def _d(self):
        if False:
            while True:
                i = 10
        return

class E:

    def _e(self):
        if False:
            print('Hello World!')
        return None