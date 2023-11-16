class SingleLineDocstrings:
    """ Double quotes single line class docstring """
    ' Not a docstring '

    def foo(self, bar='not a docstring'):
        if False:
            i = 10
            return i + 15
        ' Double quotes single line method docstring'
        pass

    class Nested(foo()[:]):
        """ inline docstring """
        pass