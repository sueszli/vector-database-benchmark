try:

    def f():
        if False:
            for i in range(10):
                print('nop')
        pass
finally:

    def g():
        if False:
            while True:
                i = 10
        pass