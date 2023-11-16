from pylint.checkers import BaseChecker
from pylint.interfaces import IAstroidChecker
from astroid.exceptions import InferenceError
import astroid

def register(linter):
    if False:
        print('Hello World!')
    linter.register_checker(ConditionalImports(linter))

class ConditionalImports(BaseChecker):
    __implements__ = (IAstroidChecker,)
    name = 'must-catch-import-error'
    msgs = {'C9997': ('Importing this module must catch ImportError.', 'must-catch-import-error', 'Importing this module must catch ImportError.')}

    def visit_import(self, node):
        if False:
            print('Hello World!')
        names = [name[0] for name in node.names]
        if 'chalice.cli.filewatch.eventbased' in names:
            if not self._is_in_try_except_import_error(node):
                self.add_message('must-catch-import-error', node=node)
                return

    def visit_importfrom(self, node):
        if False:
            print('Hello World!')
        if node.modname == 'chalice.cli.filewatch.eventbased':
            names = [name[0] for name in node.names]
            if 'WatchdogWorkerProcess' in names:
                if not self._is_in_try_except_import_error(node):
                    self.add_message('must-catch-import-error', node=node)
                    return

    def _is_in_try_except_import_error(self, node):
        if False:
            print('Hello World!')
        if not isinstance(node.parent, astroid.TryExcept):
            return False
        caught_exceptions = [handler.type.name for handler in node.parent.handlers]
        if 'ImportError' not in caught_exceptions:
            return False
        return True