def is_iterable(x):
    if False:
        while True:
            i = 10
    'An implementation independent way of checking for iterables'
    try:
        iter(x)
    except TypeError:
        return False
    else:
        return True