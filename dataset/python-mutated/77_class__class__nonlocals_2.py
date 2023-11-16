class Outer:

    def f(self):
        if False:
            i = 10
            return i + 15

        class Inner:
            nonlocal __class__
            __class__ = 42

            def f():
                if False:
                    for i in range(10):
                        print('nop')
                __class__