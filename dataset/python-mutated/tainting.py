def foo():
    if False:
        print('Hello World!')
    a = source()
    if True:
        b = a
        b = sanitize()
    else:
        b = a
    sink(b)