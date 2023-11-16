class A:

    def foo():
        if False:
            return 10
        return 1

class A:

    def foo():
        if False:
            i = 10
            return i + 15
        return 1

class A(B):

    def foo():
        if False:
            while True:
                i = 10
        return 1