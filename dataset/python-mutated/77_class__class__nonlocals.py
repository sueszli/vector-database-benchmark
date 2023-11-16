class Outer:

    class Inner:
        nonlocal __class__
        __class__ = 42

        def f():
            if False:
                while True:
                    i = 10
            __class__