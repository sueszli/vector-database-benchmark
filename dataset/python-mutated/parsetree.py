"""defines the parse tree components for Mako templates."""
import re
from mako import ast
from mako import exceptions
from mako import filters
from mako import util

class Node:
    """base class for a Node in the parse tree."""

    def __init__(self, source, lineno, pos, filename):
        if False:
            print('Hello World!')
        self.source = source
        self.lineno = lineno
        self.pos = pos
        self.filename = filename

    @property
    def exception_kwargs(self):
        if False:
            print('Hello World!')
        return {'source': self.source, 'lineno': self.lineno, 'pos': self.pos, 'filename': self.filename}

    def get_children(self):
        if False:
            i = 10
            return i + 15
        return []

    def accept_visitor(self, visitor):
        if False:
            return 10

        def traverse(node):
            if False:
                print('Hello World!')
            for n in node.get_children():
                n.accept_visitor(visitor)
        method = getattr(visitor, 'visit' + self.__class__.__name__, traverse)
        method(self)

class TemplateNode(Node):
    """a 'container' node that stores the overall collection of nodes."""

    def __init__(self, filename):
        if False:
            i = 10
            return i + 15
        super().__init__('', 0, 0, filename)
        self.nodes = []
        self.page_attributes = {}

    def get_children(self):
        if False:
            for i in range(10):
                print('nop')
        return self.nodes

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return 'TemplateNode(%s, %r)' % (util.sorted_dict_repr(self.page_attributes), self.nodes)

class ControlLine(Node):
    """defines a control line, a line-oriented python line or end tag.

    e.g.::

        % if foo:
            (markup)
        % endif

    """
    has_loop_context = False

    def __init__(self, keyword, isend, text, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.text = text
        self.keyword = keyword
        self.isend = isend
        self.is_primary = keyword in ['for', 'if', 'while', 'try', 'with']
        self.nodes = []
        if self.isend:
            self._declared_identifiers = []
            self._undeclared_identifiers = []
        else:
            code = ast.PythonFragment(text, **self.exception_kwargs)
            self._declared_identifiers = code.declared_identifiers
            self._undeclared_identifiers = code.undeclared_identifiers

    def get_children(self):
        if False:
            for i in range(10):
                print('nop')
        return self.nodes

    def declared_identifiers(self):
        if False:
            return 10
        return self._declared_identifiers

    def undeclared_identifiers(self):
        if False:
            print('Hello World!')
        return self._undeclared_identifiers

    def is_ternary(self, keyword):
        if False:
            while True:
                i = 10
        'return true if the given keyword is a ternary keyword\n        for this ControlLine'
        cases = {'if': {'else', 'elif'}, 'try': {'except', 'finally'}, 'for': {'else'}}
        return keyword in cases.get(self.keyword, set())

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'ControlLine(%r, %r, %r, %r)' % (self.keyword, self.text, self.isend, (self.lineno, self.pos))

class Text(Node):
    """defines plain text in the template."""

    def __init__(self, content, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.content = content

    def __repr__(self):
        if False:
            return 10
        return 'Text(%r, %r)' % (self.content, (self.lineno, self.pos))

class Code(Node):
    """defines a Python code block, either inline or module level.

    e.g.::

        inline:
        <%
            x = 12
        %>

        module level:
        <%!
            import logger
        %>

    """

    def __init__(self, text, ismodule, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.text = text
        self.ismodule = ismodule
        self.code = ast.PythonCode(text, **self.exception_kwargs)

    def declared_identifiers(self):
        if False:
            i = 10
            return i + 15
        return self.code.declared_identifiers

    def undeclared_identifiers(self):
        if False:
            while True:
                i = 10
        return self.code.undeclared_identifiers

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return 'Code(%r, %r, %r)' % (self.text, self.ismodule, (self.lineno, self.pos))

class Comment(Node):
    """defines a comment line.

    # this is a comment

    """

    def __init__(self, text, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.text = text

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'Comment(%r, %r)' % (self.text, (self.lineno, self.pos))

class Expression(Node):
    """defines an inline expression.

    ${x+y}

    """

    def __init__(self, text, escapes, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.text = text
        self.escapes = escapes
        self.escapes_code = ast.ArgumentList(escapes, **self.exception_kwargs)
        self.code = ast.PythonCode(text, **self.exception_kwargs)

    def declared_identifiers(self):
        if False:
            i = 10
            return i + 15
        return []

    def undeclared_identifiers(self):
        if False:
            while True:
                i = 10
        return self.code.undeclared_identifiers.union(self.escapes_code.undeclared_identifiers.difference(filters.DEFAULT_ESCAPES)).difference(self.code.declared_identifiers)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'Expression(%r, %r, %r)' % (self.text, self.escapes_code.args, (self.lineno, self.pos))

class _TagMeta(type):
    """metaclass to allow Tag to produce a subclass according to
    its keyword"""
    _classmap = {}

    def __init__(cls, clsname, bases, dict_):
        if False:
            print('Hello World!')
        if getattr(cls, '__keyword__', None) is not None:
            cls._classmap[cls.__keyword__] = cls
        super().__init__(clsname, bases, dict_)

    def __call__(cls, keyword, attributes, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if ':' in keyword:
            (ns, defname) = keyword.split(':')
            return type.__call__(CallNamespaceTag, ns, defname, attributes, **kwargs)
        try:
            cls = _TagMeta._classmap[keyword]
        except KeyError:
            raise exceptions.CompileException("No such tag: '%s'" % keyword, source=kwargs['source'], lineno=kwargs['lineno'], pos=kwargs['pos'], filename=kwargs['filename'])
        return type.__call__(cls, keyword, attributes, **kwargs)

class Tag(Node, metaclass=_TagMeta):
    """abstract base class for tags.

    e.g.::

        <%sometag/>

        <%someothertag>
            stuff
        </%someothertag>

    """
    __keyword__ = None

    def __init__(self, keyword, attributes, expressions, nonexpressions, required, **kwargs):
        if False:
            i = 10
            return i + 15
        'construct a new Tag instance.\n\n        this constructor not called directly, and is only called\n        by subclasses.\n\n        :param keyword: the tag keyword\n\n        :param attributes: raw dictionary of attribute key/value pairs\n\n        :param expressions: a set of identifiers that are legal attributes,\n         which can also contain embedded expressions\n\n        :param nonexpressions: a set of identifiers that are legal\n         attributes, which cannot contain embedded expressions\n\n        :param \\**kwargs:\n         other arguments passed to the Node superclass (lineno, pos)\n\n        '
        super().__init__(**kwargs)
        self.keyword = keyword
        self.attributes = attributes
        self._parse_attributes(expressions, nonexpressions)
        missing = [r for r in required if r not in self.parsed_attributes]
        if len(missing):
            raise exceptions.CompileException('Missing attribute(s): %s' % ','.join((repr(m) for m in missing)), **self.exception_kwargs)
        self.parent = None
        self.nodes = []

    def is_root(self):
        if False:
            i = 10
            return i + 15
        return self.parent is None

    def get_children(self):
        if False:
            return 10
        return self.nodes

    def _parse_attributes(self, expressions, nonexpressions):
        if False:
            for i in range(10):
                print('nop')
        undeclared_identifiers = set()
        self.parsed_attributes = {}
        for key in self.attributes:
            if key in expressions:
                expr = []
                for x in re.compile('(\\${.+?})', re.S).split(self.attributes[key]):
                    m = re.compile('^\\${(.+?)}$', re.S).match(x)
                    if m:
                        code = ast.PythonCode(m.group(1).rstrip(), **self.exception_kwargs)
                        undeclared_identifiers = undeclared_identifiers.union(code.undeclared_identifiers)
                        expr.append('(%s)' % m.group(1))
                    elif x:
                        expr.append(repr(x))
                self.parsed_attributes[key] = ' + '.join(expr) or repr('')
            elif key in nonexpressions:
                if re.search('\\${.+?}', self.attributes[key]):
                    raise exceptions.CompileException("Attribute '%s' in tag '%s' does not allow embedded expressions" % (key, self.keyword), **self.exception_kwargs)
                self.parsed_attributes[key] = repr(self.attributes[key])
            else:
                raise exceptions.CompileException("Invalid attribute for tag '%s': '%s'" % (self.keyword, key), **self.exception_kwargs)
        self.expression_undeclared_identifiers = undeclared_identifiers

    def declared_identifiers(self):
        if False:
            print('Hello World!')
        return []

    def undeclared_identifiers(self):
        if False:
            i = 10
            return i + 15
        return self.expression_undeclared_identifiers

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '%s(%r, %s, %r, %r)' % (self.__class__.__name__, self.keyword, util.sorted_dict_repr(self.attributes), (self.lineno, self.pos), self.nodes)

class IncludeTag(Tag):
    __keyword__ = 'include'

    def __init__(self, keyword, attributes, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(keyword, attributes, ('file', 'import', 'args'), (), ('file',), **kwargs)
        self.page_args = ast.PythonCode('__DUMMY(%s)' % attributes.get('args', ''), **self.exception_kwargs)

    def declared_identifiers(self):
        if False:
            for i in range(10):
                print('nop')
        return []

    def undeclared_identifiers(self):
        if False:
            print('Hello World!')
        identifiers = self.page_args.undeclared_identifiers.difference({'__DUMMY'}).difference(self.page_args.declared_identifiers)
        return identifiers.union(super().undeclared_identifiers())

class NamespaceTag(Tag):
    __keyword__ = 'namespace'

    def __init__(self, keyword, attributes, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(keyword, attributes, ('file',), ('name', 'inheritable', 'import', 'module'), (), **kwargs)
        self.name = attributes.get('name', '__anon_%s' % hex(abs(id(self))))
        if 'name' not in attributes and 'import' not in attributes:
            raise exceptions.CompileException("'name' and/or 'import' attributes are required for <%namespace>", **self.exception_kwargs)
        if 'file' in attributes and 'module' in attributes:
            raise exceptions.CompileException("<%namespace> may only have one of 'file' or 'module'", **self.exception_kwargs)

    def declared_identifiers(self):
        if False:
            print('Hello World!')
        return []

class TextTag(Tag):
    __keyword__ = 'text'

    def __init__(self, keyword, attributes, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(keyword, attributes, (), 'filter', (), **kwargs)
        self.filter_args = ast.ArgumentList(attributes.get('filter', ''), **self.exception_kwargs)

    def undeclared_identifiers(self):
        if False:
            i = 10
            return i + 15
        return self.filter_args.undeclared_identifiers.difference(filters.DEFAULT_ESCAPES.keys()).union(self.expression_undeclared_identifiers)

class DefTag(Tag):
    __keyword__ = 'def'

    def __init__(self, keyword, attributes, **kwargs):
        if False:
            return 10
        expressions = ['buffered', 'cached'] + [c for c in attributes if c.startswith('cache_')]
        super().__init__(keyword, attributes, expressions, ('name', 'filter', 'decorator'), ('name',), **kwargs)
        name = attributes['name']
        if re.match('^[\\w_]+$', name):
            raise exceptions.CompileException('Missing parenthesis in %def', **self.exception_kwargs)
        self.function_decl = ast.FunctionDecl('def ' + name + ':pass', **self.exception_kwargs)
        self.name = self.function_decl.funcname
        self.decorator = attributes.get('decorator', '')
        self.filter_args = ast.ArgumentList(attributes.get('filter', ''), **self.exception_kwargs)
    is_anonymous = False
    is_block = False

    @property
    def funcname(self):
        if False:
            print('Hello World!')
        return self.function_decl.funcname

    def get_argument_expressions(self, **kw):
        if False:
            print('Hello World!')
        return self.function_decl.get_argument_expressions(**kw)

    def declared_identifiers(self):
        if False:
            for i in range(10):
                print('nop')
        return self.function_decl.allargnames

    def undeclared_identifiers(self):
        if False:
            while True:
                i = 10
        res = []
        for c in self.function_decl.defaults:
            res += list(ast.PythonCode(c, **self.exception_kwargs).undeclared_identifiers)
        return set(res).union(self.filter_args.undeclared_identifiers.difference(filters.DEFAULT_ESCAPES.keys())).union(self.expression_undeclared_identifiers).difference(self.function_decl.allargnames)

class BlockTag(Tag):
    __keyword__ = 'block'

    def __init__(self, keyword, attributes, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        expressions = ['buffered', 'cached', 'args'] + [c for c in attributes if c.startswith('cache_')]
        super().__init__(keyword, attributes, expressions, ('name', 'filter', 'decorator'), (), **kwargs)
        name = attributes.get('name')
        if name and (not re.match('^[\\w_]+$', name)):
            raise exceptions.CompileException('%block may not specify an argument signature', **self.exception_kwargs)
        if not name and attributes.get('args', None):
            raise exceptions.CompileException('Only named %blocks may specify args', **self.exception_kwargs)
        self.body_decl = ast.FunctionArgs(attributes.get('args', ''), **self.exception_kwargs)
        self.name = name
        self.decorator = attributes.get('decorator', '')
        self.filter_args = ast.ArgumentList(attributes.get('filter', ''), **self.exception_kwargs)
    is_block = True

    @property
    def is_anonymous(self):
        if False:
            for i in range(10):
                print('nop')
        return self.name is None

    @property
    def funcname(self):
        if False:
            i = 10
            return i + 15
        return self.name or '__M_anon_%d' % (self.lineno,)

    def get_argument_expressions(self, **kw):
        if False:
            return 10
        return self.body_decl.get_argument_expressions(**kw)

    def declared_identifiers(self):
        if False:
            while True:
                i = 10
        return self.body_decl.allargnames

    def undeclared_identifiers(self):
        if False:
            while True:
                i = 10
        return self.filter_args.undeclared_identifiers.difference(filters.DEFAULT_ESCAPES.keys()).union(self.expression_undeclared_identifiers)

class CallTag(Tag):
    __keyword__ = 'call'

    def __init__(self, keyword, attributes, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(keyword, attributes, 'args', ('expr',), ('expr',), **kwargs)
        self.expression = attributes['expr']
        self.code = ast.PythonCode(self.expression, **self.exception_kwargs)
        self.body_decl = ast.FunctionArgs(attributes.get('args', ''), **self.exception_kwargs)

    def declared_identifiers(self):
        if False:
            i = 10
            return i + 15
        return self.code.declared_identifiers.union(self.body_decl.allargnames)

    def undeclared_identifiers(self):
        if False:
            return 10
        return self.code.undeclared_identifiers.difference(self.code.declared_identifiers)

class CallNamespaceTag(Tag):

    def __init__(self, namespace, defname, attributes, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(namespace + ':' + defname, attributes, tuple(attributes.keys()) + ('args',), (), (), **kwargs)
        self.expression = '%s.%s(%s)' % (namespace, defname, ','.join(('%s=%s' % (k, v) for (k, v) in self.parsed_attributes.items() if k != 'args')))
        self.code = ast.PythonCode(self.expression, **self.exception_kwargs)
        self.body_decl = ast.FunctionArgs(attributes.get('args', ''), **self.exception_kwargs)

    def declared_identifiers(self):
        if False:
            print('Hello World!')
        return self.code.declared_identifiers.union(self.body_decl.allargnames)

    def undeclared_identifiers(self):
        if False:
            for i in range(10):
                print('nop')
        return self.code.undeclared_identifiers.difference(self.code.declared_identifiers)

class InheritTag(Tag):
    __keyword__ = 'inherit'

    def __init__(self, keyword, attributes, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(keyword, attributes, ('file',), (), ('file',), **kwargs)

class PageTag(Tag):
    __keyword__ = 'page'

    def __init__(self, keyword, attributes, **kwargs):
        if False:
            i = 10
            return i + 15
        expressions = ['cached', 'args', 'expression_filter', 'enable_loop'] + [c for c in attributes if c.startswith('cache_')]
        super().__init__(keyword, attributes, expressions, (), (), **kwargs)
        self.body_decl = ast.FunctionArgs(attributes.get('args', ''), **self.exception_kwargs)
        self.filter_args = ast.ArgumentList(attributes.get('expression_filter', ''), **self.exception_kwargs)

    def declared_identifiers(self):
        if False:
            return 10
        return self.body_decl.allargnames