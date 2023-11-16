class abstractclassmethod(object):
    """A Python 3.2 STORE_LOCALS bug
    """

    def __init__(self, callable):
        if False:
            return 10
        callable.__isabstractmethod__ = True