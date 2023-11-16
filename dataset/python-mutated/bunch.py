class SimpleBunch(dict):
    """Container object for datasets: dictionnary-like object that
    exposes its keys as attributes.
    """

    def __init__(self, **kwargs):
        if False:
            return 10
        dict.__init__(self, kwargs)
        self.__dict__ = self