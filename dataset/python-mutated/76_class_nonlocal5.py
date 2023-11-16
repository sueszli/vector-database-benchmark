def top_method(self):
    if False:
        print('Hello World!')

    def outer():
        if False:
            i = 10
            return i + 15

        class Test:

            def actual_global(self):
                if False:
                    i = 10
                    return i + 15
                return str('global')

            def str(self):
                if False:
                    while True:
                        i = 10
                return str(self)