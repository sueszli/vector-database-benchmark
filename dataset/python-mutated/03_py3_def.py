def x0(*, file='sample_file'):
    if False:
        i = 10
        return i + 15
    pass

def x1(a, *, file=None):
    if False:
        return 10
    pass

def x2(a=5, *, file='filex2'):
    if False:
        print('Hello World!')
    pass

def x3(a=5, *, file='filex2', stuff=None):
    if False:
        return 10
    pass