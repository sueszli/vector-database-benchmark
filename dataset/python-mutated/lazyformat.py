class lazyformat:
    """A format string that isn't evaluated until it's needed."""

    def __init__(self, format_string, *args):
        if False:
            print('Hello World!')
        self.__format_string = format_string
        self.__args = args

    def __str__(self):
        if False:
            return 10
        return self.__format_string % self.__args

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return isinstance(other, lazyformat) and self.__format_string == other.__format_string and (self.__args == other.__args)

    def __ne__(self, other):
        if False:
            return 10
        return not self.__eq__(other)

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return hash(self.__format_string)