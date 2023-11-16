def f():
    if False:
        for i in range(10):
            print('nop')
    with foo as bar:
        a