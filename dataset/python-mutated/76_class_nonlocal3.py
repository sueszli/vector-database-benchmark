def testNonLocalClass(self):
    if False:
        return 10

    def f(x):
        if False:
            while True:
                i = 10

        class c:
            nonlocal x
            x += 1

            def get(self):
                if False:
                    while True:
                        i = 10
                return x
        return c()