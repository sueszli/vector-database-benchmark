from trashcli.compat import Protocol

class IntGenerator(Protocol):

    def new_int(self, min, max):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError