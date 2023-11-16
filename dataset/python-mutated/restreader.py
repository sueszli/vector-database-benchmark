import functools
from robot.errors import DataError
try:
    from docutils.core import publish_doctree
    from docutils.parsers.rst import directives
    from docutils.parsers.rst import roles
    from docutils.parsers.rst.directives import register_directive
    from docutils.parsers.rst.directives.body import CodeBlock
    from docutils.parsers.rst.directives.misc import Include
except ImportError:
    raise DataError("Using reStructuredText test data requires having 'docutils' module version 0.9 or newer installed.")

class RobotDataStorage:

    def __init__(self, doctree):
        if False:
            for i in range(10):
                print('nop')
        if not hasattr(doctree, '_robot_data'):
            doctree._robot_data = []
        self._robot_data = doctree._robot_data

    def add_data(self, rows):
        if False:
            i = 10
            return i + 15
        self._robot_data.extend(rows)

    def get_data(self):
        if False:
            return 10
        return '\n'.join(self._robot_data)

    def has_data(self):
        if False:
            print('Hello World!')
        return bool(self._robot_data)

class RobotCodeBlock(CodeBlock):

    def run(self):
        if False:
            while True:
                i = 10
        if 'robotframework' in self.arguments:
            store = RobotDataStorage(self.state_machine.document)
            store.add_data(self.content)
        return []
register_directive('code', RobotCodeBlock)
register_directive('code-block', RobotCodeBlock)
register_directive('sourcecode', RobotCodeBlock)
relevant_directives = (RobotCodeBlock, Include)

@functools.wraps(directives.directive)
def directive(*args, **kwargs):
    if False:
        return 10
    (directive_class, messages) = directive.__wrapped__(*args, **kwargs)
    if directive_class not in relevant_directives:
        directive_class = lambda *args, **kwargs: []
    return (directive_class, messages)

@functools.wraps(roles.role)
def role(*args, **kwargs):
    if False:
        print('Hello World!')
    role_function = role.__wrapped__(*args, **kwargs)
    if role_function is None:
        role_function = (lambda *args, **kwargs: [], [])
    return role_function
directives.directive = directive
roles.role = role

def read_rest_data(rstfile):
    if False:
        for i in range(10):
            print('nop')
    doctree = publish_doctree(rstfile.read(), source_path=rstfile.name, settings_overrides={'input_encoding': 'UTF-8', 'report_level': 4})
    store = RobotDataStorage(doctree)
    return store.get_data()