def foo():
    if False:
        print('Hello World!')
    global bar

    def bar():
        if False:
            i = 10
            return i + 15
        pass