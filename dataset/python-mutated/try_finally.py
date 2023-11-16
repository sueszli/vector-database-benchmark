def foo():
    if False:
        for i in range(10):
            print('nop')
    try:
        raise Foo()
    finally:
        sink(source)