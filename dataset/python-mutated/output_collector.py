from six import StringIO
from .support.asserts import assert_equals_with_unidiff

class OutputCollector:

    def __init__(self):
        if False:
            return 10
        self.stream = StringIO()
        self.getvalue = self.stream.getvalue

    def write(self, data):
        if False:
            return 10
        self.stream.write(data)

    def should_be(self, expected):
        if False:
            i = 10
            return i + 15
        assert_equals_with_unidiff(expected, self.output())

    def output(self):
        if False:
            return 10
        return self.stream.getvalue()