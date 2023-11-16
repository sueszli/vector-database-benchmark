def foo():
    if False:
        while True:
            i = 10
    bar()
    x = requests.get('foo')
    bar()
    if True:
        render(o.format(x))