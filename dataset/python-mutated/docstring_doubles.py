"""
Double quotes multiline module docstring
"""
'\nthis is not a docstring\n'
l = []

class Cls:
    """
    Double quotes multiline class docstring
    """
    '\n    this is not a docstring\n    '

    def f(self, bar='\n        definitely not a docstring', val=l[Cls():3]):
        if False:
            return 10
        '\n        Double quotes multiline function docstring\n        '
        some_expression = 'hello world'
        '\n        this is not a docstring\n        '
        if l:
            "\n            Looks like a docstring, but in reality it isn't - only modules, classes and functions\n            "
            pass