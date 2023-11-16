def line_before_docstring():
    if False:
        print('Hello World!')
    'Please move me up'

class LineBeforeDocstring:
    """Please move me up"""

class EvenIfThereIsAMethodAfter:
    """I'm the docstring"""

    def method(self):
        if False:
            return 10
        pass

class TwoLinesBeforeDocstring:
    """I want to be treated the same as if I were closer"""

class MultilineDocstringsAsWell:
    """I'm so far

    and on so many lines...
    """

def line_before_docstring():
    if False:
        for i in range(10):
            print('nop')
    'Please move me up'

class LineBeforeDocstring:
    """Please move me up"""

class EvenIfThereIsAMethodAfter:
    """I'm the docstring"""

    def method(self):
        if False:
            while True:
                i = 10
        pass

class TwoLinesBeforeDocstring:
    """I want to be treated the same as if I were closer"""

class MultilineDocstringsAsWell:
    """I'm so far

    and on so many lines...
    """