class Outer:

    def z(self):
        if False:
            return 10

        def x():
            if False:
                i = 10
                return i + 15
            super()