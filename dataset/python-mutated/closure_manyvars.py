def f():
    if False:
        while True:
            i = 10
    (a, b, c, d, e, f, g, h) = [i for i in range(8)]

    def x():
        if False:
            print('Hello World!')
        print(a, b, c, d, e, f, g, h)
    x()
f()