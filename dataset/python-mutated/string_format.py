"""Ansible specific pylint plugin for checking format string usage."""
from __future__ import annotations
import astroid
try:
    from pylint.interfaces import IAstroidChecker
except ImportError:

    class IAstroidChecker:
        """Backwards compatibility for 2.x / 3.x support."""
try:
    from pylint.checkers.utils import check_messages
except ImportError:
    from pylint.checkers.utils import only_required_for_messages as check_messages
from pylint.checkers import BaseChecker
from pylint.checkers import utils
MSGS = {'E9305': ('disabled', 'ansible-format-automatic-specification', 'disabled'), 'E9390': ('bytes object has no .format attribute', 'ansible-no-format-on-bytestring', 'Used when a bytestring was used as a PEP 3101 format string as Python3 bytestrings do not have a .format attribute', {'minversion': (3, 0)})}

class AnsibleStringFormatChecker(BaseChecker):
    """Checks string formatting operations to ensure that the format string
    is valid and the arguments match the format string.
    """
    __implements__ = (IAstroidChecker,)
    name = 'string'
    msgs = MSGS

    @check_messages(*MSGS.keys())
    def visit_call(self, node):
        if False:
            print('Hello World!')
        'Visit a call node.'
        func = utils.safe_infer(node.func)
        if isinstance(func, astroid.BoundMethod) and isinstance(func.bound, astroid.Instance) and (func.bound.name in ('str', 'unicode', 'bytes')):
            if func.name == 'format':
                self._check_new_format(node, func)

    def _check_new_format(self, node, func):
        if False:
            for i in range(10):
                print('nop')
        ' Check the new string formatting '
        if isinstance(node.func, astroid.Attribute) and (not isinstance(node.func.expr, astroid.Const)):
            return
        try:
            strnode = next(func.bound.infer())
        except astroid.InferenceError:
            return
        if not isinstance(strnode, astroid.Const):
            return
        if isinstance(strnode.value, bytes):
            self.add_message('ansible-no-format-on-bytestring', node=node)
            return

def register(linter):
    if False:
        i = 10
        return i + 15
    'required method to auto register this checker '
    linter.register_checker(AnsibleStringFormatChecker(linter))