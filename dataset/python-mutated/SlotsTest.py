class W1(object):

    def __init__(self):
        if False:
            return 10
        self.__hidden = 5

class W2(object):
    __slots__ = ['__hidden']

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.__hidden = 5

class _W1(object):

    def __init__(self):
        if False:
            return 10
        self.__hidden = 5

class _W2(object):
    __slots__ = ['__hidden']

    def __init__(self):
        if False:
            print('Hello World!')
        self.__hidden = 5

class a_W1(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.__hidden = 5

class a_W2(object):
    __slots__ = ['__hidden']

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.__hidden = 5

class W1_(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.__hidden = 5

class W2_(object):
    __slots__ = ['__hidden']

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.__hidden = 5
for w in (W1, W2, _W1, _W2, a_W1, a_W2, W1_, W2_):
    try:
        print(w)
        print(dir(w))
        a = w()
    except AttributeError:
        print('bug in %s' % w)