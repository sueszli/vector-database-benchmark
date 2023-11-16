from foo import f, dialect, args, kwds, reader
f(*[])
x = reader(f, dialect, *args, **kwds)

def cmp_to_key(mycmp):
    if False:
        print('Hello World!')

    class K(object):

        def __ge__():
            if False:
                return 10
            return mycmp()
    return

def cmp2_to_key(mycmp):
    if False:
        for i in range(10):
            print('nop')

    class K2(object):

        def __ge__():
            if False:
                print('Hello World!')
            return 5
    return