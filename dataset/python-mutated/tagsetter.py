from typing import Sequence, TYPE_CHECKING
from .visitor import SuiteVisitor
if TYPE_CHECKING:
    from .keyword import Keyword
    from .testcase import TestCase
    from .testsuite import TestSuite

class TagSetter(SuiteVisitor):

    def __init__(self, add: 'Sequence[str]|str'=(), remove: 'Sequence[str]|str'=()):
        if False:
            return 10
        self.add = add
        self.remove = remove

    def start_suite(self, suite: 'TestSuite'):
        if False:
            print('Hello World!')
        return bool(self)

    def visit_test(self, test: 'TestCase'):
        if False:
            print('Hello World!')
        test.tags.add(self.add)
        test.tags.remove(self.remove)

    def visit_keyword(self, keyword: 'Keyword'):
        if False:
            print('Hello World!')
        pass

    def __bool__(self):
        if False:
            i = 10
            return i + 15
        return bool(self.add or self.remove)