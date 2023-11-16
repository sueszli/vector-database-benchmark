class UniqueIdentifier:
    """A factory for sentinel objects with nice reprs."""

    def __init__(self, identifier):
        if False:
            while True:
                i = 10
        self.identifier = identifier

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return self.identifier
infer = ...
not_set = UniqueIdentifier('not_set')