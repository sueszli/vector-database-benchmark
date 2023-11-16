class Foo:

    def test_various___class___pathologies(self):
        if False:
            return 10

        class X:

            def f(self):
                if False:
                    for i in range(10):
                        print('nop')
                return super().f()
            __class__ = 413
        x = X()

        class X:
            x = __class__

            def f():
                if False:
                    for i in range(10):
                        print('nop')
                __class__

        class X:
            global __class__
            __class__ = 42

            def f():
                if False:
                    print('Hello World!')
                __class__