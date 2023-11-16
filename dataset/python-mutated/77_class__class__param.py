class Outer:

    def x(self):
        if False:
            return 10

        def f(__class__):
            if False:
                return 10
            __class__