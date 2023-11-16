def a():
    if False:
        for i in range(10):
            print('nop')
    del y

    def b():
        if False:
            return 10
        return y