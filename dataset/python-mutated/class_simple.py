class Base(object):

    class Nested:

        def foo():
            if False:
                while True:
                    i = 10
            pass

class X(Base.Nested):
    pass
X().foo()
X().bar()