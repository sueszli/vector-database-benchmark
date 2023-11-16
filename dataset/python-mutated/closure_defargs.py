def f():
    if False:
        i = 10
        return i + 15
    a = 1

    def bar(b=10, c=20):
        if False:
            for i in range(10):
                print('nop')
        print(a + b + c)
    bar()
    bar(2)
    bar(2, 3)
print(f())