class DynamicLibraryWithoutGetKwArgsAndDoc:
    """Library doc set in class."""

    def __init__(self, doc=None):
        if False:
            i = 10
            return i + 15
        'Static __init__ doc.'
        if doc:
            self.__doc__ = doc

    def get_keyword_names(self):
        if False:
            i = 10
            return i + 15
        return ['Keyword']

    def run_keyword(self, name, args):
        if False:
            while True:
                i = 10
        pass