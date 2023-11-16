"""Print an AST tree in a form more readable than ast.dump."""
import gast
import six

class PrettyPrinter(gast.NodeVisitor):
    """Print AST nodes."""

    def __init__(self, color, noanno):
        if False:
            print('Hello World!')
        self.indent_lvl = 0
        self.result = ''
        self.color = color
        self.noanno = noanno

    def _color(self, string, color, attrs=None):
        if False:
            print('Hello World!')
        return string

    def _type(self, node):
        if False:
            i = 10
            return i + 15
        return self._color(node.__class__.__name__, None, ['bold'])

    def _field(self, name):
        if False:
            print('Hello World!')
        return self._color(name, 'blue')

    def _value(self, name):
        if False:
            print('Hello World!')
        return self._color(name, 'magenta')

    def _warning(self, name):
        if False:
            while True:
                i = 10
        return self._color(name, 'red')

    def _indent(self):
        if False:
            return 10
        return self._color('| ' * self.indent_lvl, None, ['dark'])

    def _print(self, s):
        if False:
            for i in range(10):
                print('nop')
        self.result += s
        self.result += '\n'

    def generic_visit(self, node, name=None):
        if False:
            print('Hello World!')
        if isinstance(node, str):
            if name:
                self._print('%s%s="%s"' % (self._indent(), name, node))
            else:
                self._print('%s"%s"' % (self._indent(), node))
            return
        if node._fields:
            cont = ':'
        else:
            cont = '()'
        if name:
            self._print('%s%s=%s%s' % (self._indent(), self._field(name), self._type(node), cont))
        else:
            self._print('%s%s%s' % (self._indent(), self._type(node), cont))
        self.indent_lvl += 1
        for f in node._fields:
            if self.noanno and f.startswith('__'):
                continue
            if not hasattr(node, f):
                self._print('%s%s' % (self._indent(), self._warning('%s=<unset>' % f)))
                continue
            v = getattr(node, f)
            if isinstance(v, list):
                if v:
                    self._print('%s%s=[' % (self._indent(), self._field(f)))
                    self.indent_lvl += 1
                    for n in v:
                        if n is not None:
                            self.generic_visit(n)
                        else:
                            self._print('%sNone' % self._indent())
                    self.indent_lvl -= 1
                    self._print('%s]' % self._indent())
                else:
                    self._print('%s%s=[]' % (self._indent(), self._field(f)))
            elif isinstance(v, tuple):
                if v:
                    self._print('%s%s=(' % (self._indent(), self._field(f)))
                    self.indent_lvl += 1
                    for n in v:
                        if n is not None:
                            self.generic_visit(n)
                        else:
                            self._print('%sNone' % self._indent())
                    self.indent_lvl -= 1
                    self._print('%s)' % self._indent())
                else:
                    self._print('%s%s=()' % (self._indent(), self._field(f)))
            elif isinstance(v, gast.AST):
                self.generic_visit(v, f)
            elif isinstance(v, six.binary_type):
                self._print('%s%s=%s' % (self._indent(), self._field(f), self._value('b"%s"' % v)))
            elif isinstance(v, six.text_type):
                self._print('%s%s=%s' % (self._indent(), self._field(f), self._value('u"%s"' % v)))
            else:
                self._print('%s%s=%s' % (self._indent(), self._field(f), self._value(v)))
        self.indent_lvl -= 1

def fmt(node, color=True, noanno=False):
    if False:
        for i in range(10):
            print('nop')
    printer = PrettyPrinter(color, noanno)
    if isinstance(node, (list, tuple)):
        for n in node:
            printer.visit(n)
    else:
        printer.visit(node)
    return printer.result