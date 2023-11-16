class IDBox(object):
    """A wrapper that hashs to the id of the underlying object and compares
    equality on the id of the underlying.

    Parameters
    ----------
    ob : any
        The object to wrap.

    Attributes
    ----------
    ob : any
        The object being wrapped.

    Notes
    -----
    This is useful for storing non-hashable values in a set or dict.
    """

    def __init__(self, ob):
        if False:
            while True:
                i = 10
        self.ob = ob

    def __hash__(self):
        if False:
            while True:
                i = 10
        return id(self)

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        if not isinstance(other, IDBox):
            return NotImplemented
        return id(self.ob) == id(other.ob)