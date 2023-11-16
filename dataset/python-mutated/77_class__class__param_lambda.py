class Outer:

    def x(self):
        if False:
            for i in range(10):
                print('nop')

        def f(__class__):
            if False:
                print('Hello World!')
            lambda : __class__