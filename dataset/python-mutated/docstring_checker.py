"""DocstringChecker is used to check python doc string's style."""
import re
from collections import defaultdict
import astroid
from pylint.checkers import BaseChecker
from pylint.interfaces import IAstroidChecker

def register(linter):
    if False:
        return 10
    'Register checkers.'
    linter.register_checker(DocstringChecker(linter))

class Docstring:
    """Docstring class holds the parsed doc string elements."""

    def __init__(self):
        if False:
            print('Hello World!')
        self.d = defaultdict(list)
        self.clear()

    def clear(self):
        if False:
            for i in range(10):
                print('nop')
        self.d['Args'] = []
        self.d['Examples'] = []
        self.d['Returns'] = []
        self.d['Raises'] = []
        self.args = {}

    def get_level(self, string, indent='    '):
        if False:
            print('Hello World!')
        level = 0
        unit_size = len(indent)
        while string[:unit_size] == indent:
            string = string[unit_size:]
            level += 1
        return level

    def parse(self, doc):
        if False:
            return 10
        'parse gets sections from doc\n        Such as Args, Returns, Raises, Examples s\n        Args:\n            doc (string): is the astroid node doc string.\n        Returns:\n            True if doc is parsed successfully.\n        '
        self.clear()
        lines = doc.splitlines()
        state = ('others', -1)
        for l in lines:
            c = l.strip()
            if len(c) <= 0:
                continue
            level = self.get_level(l)
            if c.startswith('Args:'):
                state = ('Args', level)
            elif c.startswith('Returns:'):
                state = ('Returns', level)
            elif c.startswith('Raises:'):
                state = ('Raises', level)
            elif c.startswith('Examples:'):
                state = ('Examples', level)
            else:
                if level > state[1]:
                    self.d[state[0]].append(c)
                    continue
                state = ('others', -1)
                self.d[state[0]].append(c)
        self._arg_with_type()
        return True

    def get_returns(self):
        if False:
            print('Hello World!')
        return self.d['Returns']

    def get_raises(self):
        if False:
            return 10
        return self.d['Raises']

    def get_examples(self):
        if False:
            while True:
                i = 10
        return self.d['Examples']

    def _arg_with_type(self):
        if False:
            for i in range(10):
                print('nop')
        for t in self.d['Args']:
            m = re.search('([A-Za-z0-9_-]+)\\s{0,4}(\\(.+\\))\\s{0,4}:', t)
            if m:
                self.args[m.group(1)] = m.group(2)
        return self.args

class DocstringChecker(BaseChecker):
    """DosstringChecker is pylint checker to
    check docstring style.
    """
    __implements__ = (IAstroidChecker,)
    POSITIONAL_MESSAGE_ID = 'str-used-on-positional-format-argument'
    KEYWORD_MESSAGE_ID = 'str-used-on-keyword-format-argument'
    name = 'doc-string-checker'
    symbol = 'doc-string'
    priority = -1
    msgs = {'W9001': ('One line doc string on > 1 lines', symbol + '-one-line', 'Used when a short doc string is on multiple lines'), 'W9002': ('Doc string does not end with "." period', symbol + '-end-with', 'Used when a doc string does not end with a period'), 'W9003': ('All args with their types must be mentioned in doc string %s', symbol + '-with-all-args', 'Used when not all arguments are in the doc string '), 'W9005': ('Missing docstring or docstring is too short', symbol + '-missing', 'Add docstring longer >=10'), 'W9006': ('Docstring indent error, use 4 space for indent', symbol + '-indent-error', 'Use 4 space for indent'), 'W9007': ('You should add `Returns` in comments', symbol + '-with-returns', 'There should be a `Returns` section in comments'), 'W9008': ('You should add `Raises` section in comments', symbol + '-with-raises', 'There should be a `Raises` section in comments')}
    options = ()

    def visit_functiondef(self, node):
        if False:
            print('Hello World!')
        'visit_functiondef checks Function node docstring style.\n        Args:\n            node (astroid.node): The visiting node.\n        Returns:\n            True if successful other wise False.\n        '
        self.check_doc_string(node)
        if node.tolineno - node.fromlineno <= 10:
            return True
        if not node.doc:
            return True
        doc = Docstring()
        doc.parse(node.doc)
        self.all_args_in_doc(node, doc)
        self.with_returns(node, doc)
        self.with_raises(node, doc)

    def visit_module(self, node):
        if False:
            i = 10
            return i + 15
        self.check_doc_string(node)

    def visit_classdef(self, node):
        if False:
            while True:
                i = 10
        self.check_doc_string(node)

    def check_doc_string(self, node):
        if False:
            i = 10
            return i + 15
        self.missing_doc_string(node)
        self.one_line(node)
        self.has_period(node)
        self.indent_style(node)

    def missing_doc_string(self, node):
        if False:
            return 10
        if node.name.startswith('__') or node.name.startswith('_'):
            return True
        if node.tolineno - node.fromlineno <= 10:
            return True
        if node.doc is None or len(node.doc) < 10:
            self.add_message('W9005', node=node, line=node.fromlineno)
        return False

    def indent_style(self, node, indent=4):
        if False:
            i = 10
            return i + 15
        "indent_style checks docstring's indent style\n        Args:\n            node (astroid.node): The visiting node.\n            indent (int): The default indent of style\n        Returns:\n            True if successful other wise False.\n        "
        if node.doc is None:
            return True
        doc = node.doc
        lines = doc.splitlines()
        line_num = 0
        for l in lines:
            if line_num == 0:
                continue
            cur_indent = len(l) - len(l.lstrip())
            if cur_indent % indent != 0:
                self.add_message('W9006', node=node, line=node.fromlineno)
                return False
            line_num += 1
        return True

    def one_line(self, node):
        if False:
            while True:
                i = 10
        'one_line checks if docstring (len < 40) is on one line.\n        Args:\n            node (astroid.node): The node visiting.\n        Returns:\n            True if successful otherwise False.\n        '
        doc = node.doc
        if doc is None:
            return True
        if len(doc) > 40:
            return True
        elif sum((doc.find(nl) for nl in ('\n', '\r', '\n\r'))) == -3:
            return True
        else:
            self.add_message('W9001', node=node, line=node.fromlineno)
            return False
        return True

    def has_period(self, node):
        if False:
            for i in range(10):
                print('nop')
        "has_period checks if one line doc end-with '.' .\n        Args:\n            node (astroid.node): the node is visiting.\n        Returns:\n            True if successful otherwise False.\n        "
        if node.doc is None:
            return True
        if len(node.doc.splitlines()) > 1:
            return True
        if not node.doc.strip().endswith('.'):
            self.add_message('W9002', node=node, line=node.fromlineno)
            return False
        return True

    def with_raises(self, node, doc):
        if False:
            return 10
        "with_raises checks if one line doc end-with '.' .\n        Args:\n            node (astroid.node): the node is visiting.\n            doc (Docstring): Docstring object.\n        Returns:\n            True if successful otherwise False.\n        "
        find = False
        for t in node.body:
            if not isinstance(t, astroid.Raise):
                continue
            find = True
            break
        if not find:
            return True
        if len(doc.get_raises()) == 0:
            self.add_message('W9008', node=node, line=node.fromlineno)
            return False
        return True

    def with_returns(self, node, doc):
        if False:
            i = 10
            return i + 15
        'with_returns checks if docstring comments what are returned .\n        Args:\n            node (astroid.node): the node is visiting.\n            doc (Docstring): Docstring object.\n        Returns:\n            True if successful otherwise False.\n        '
        if node.name.startswith('__') or node.name.startswith('_'):
            return True
        find = False
        for t in node.body:
            if not isinstance(t, astroid.Return):
                continue
            find = True
            break
        if not find:
            return True
        if len(doc.get_returns()) == 0:
            self.add_message('W9007', node=node, line=node.fromlineno)
            return False
        return True

    def all_args_in_doc(self, node, doc):
        if False:
            i = 10
            return i + 15
        'all_args_in_doc checks if arguments are mentioned in doc\n        Args:\n            node (astroid.node): the node is visiting.\n            doc (Docstring): Docstring object\n        Returns:\n            True if successful otherwise False.\n        '
        if node.name.startswith('__') or node.name.startswith('_'):
            return True
        args = []
        for arg in node.args.get_children():
            if not isinstance(arg, astroid.AssignName) or arg.name == 'self':
                continue
            args.append(arg.name)
        if len(args) <= 0:
            return True
        parsed_args = doc.args
        args_not_documented = set(args) - set(parsed_args)
        if len(args) > 0 and len(parsed_args) <= 0:
            self.add_message('W9003', node=node, line=node.fromlineno, args=list(args_not_documented))
            return False
        for t in args:
            if t not in parsed_args:
                self.add_message('W9003', node=node, line=node.fromlineno, args=[t])
                return False
        return True