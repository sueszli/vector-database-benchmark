from robot.model import Tags
from robot.result import Keyword as KeywordResult
from robot.variables import VariableAssignment
from .arguments import ArgumentSpec
from .statusreporter import StatusReporter

class UserErrorHandler:
    """Created if creating handlers fail. Running it raises DataError.

    The idea is not to raise DataError at processing time and prevent all
    tests in affected test case file from executing. Instead, UserErrorHandler
    is created and if it is ever run DataError is raised then.
    """
    supports_embedded_arguments = False

    def __init__(self, error, name, owner=None, source=None, lineno=None):
        if False:
            while True:
                i = 10
        '\n        :param robot.errors.DataError error: Occurred error.\n        :param str name: Name of the affected keyword.\n        :param str owner: Name of the affected library or resource.\n        :param str source: Path to the source file.\n        :param int lineno: Line number of the failing keyword.\n        '
        self.error = error
        self.name = name
        self.owner = owner
        self.source = source
        self.lineno = lineno
        self.arguments = ArgumentSpec()
        self.timeout = None
        self.tags = Tags()

    @property
    def full_name(self):
        if False:
            while True:
                i = 10
        return f'{self.owner}.{self.name}' if self.owner else self.name

    @property
    def doc(self):
        if False:
            i = 10
            return i + 15
        return f'*Creating keyword failed:* {self.error}'

    @property
    def short_doc(self):
        if False:
            print('Hello World!')
        return self.doc.splitlines()[0]

    def create_runner(self, name, languages=None):
        if False:
            for i in range(10):
                print('nop')
        return self

    def run(self, kw, context, run=True):
        if False:
            return 10
        result = KeywordResult(name=self.name, owner=self.owner, args=kw.args, assign=tuple(VariableAssignment(kw.assign)), type=kw.type)
        with StatusReporter(kw, result, context, run):
            if run:
                raise self.error
    dry_run = run