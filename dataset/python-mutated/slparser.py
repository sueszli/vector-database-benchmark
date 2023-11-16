from __future__ import division, absolute_import, with_statement, print_function, unicode_literals
from renpy.compat import PY2, basestring, bchr, bord, chr, open, pystr, range, round, str, tobytes, unicode
from typing import Any, Callable, Literal
import collections
import renpy
import renpy.sl2.slast as slast
from ast import literal_eval
STYLE_PREFIXES = ['', 'insensitive_', 'hover_', 'idle_', 'activate_', 'selected_', 'selected_insensitive_', 'selected_hover_', 'selected_idle_', 'selected_activate_']
parser = None
statements = dict()
all_child_statements = []
childbearing_statements = set()

class Positional(object):
    """
    This represents a positional parameter to a function.
    """

    def __init__(self, name):
        if False:
            print('Hello World!')
        self.name = name
        if parser:
            parser.add(self)
properties = collections.defaultdict(set)

class Keyword(object):
    """
    This represents an optional keyword parameter to a function.
    """

    def __init__(self, name):
        if False:
            i = 10
            return i + 15
        self.name = name
        properties['', False].add(name)
        if parser:
            parser.add(self)

class Style(object):
    """
    This represents a style parameter to a function.
    """

    def __init__(self, name):
        if False:
            return 10
        self.name = name
        properties['', True].add(self.name)
        if parser:
            parser.add(self)

class PrefixStyle(object):
    """
    This represents a prefixed style parameter to a function.
    """

    def __init__(self, prefix, name):
        if False:
            print('Hello World!')
        self.prefix = prefix
        self.name = name
        properties[prefix, True].add(self.name)
        if parser:
            parser.add(self)
from renpy.styledata.stylesets import proxy_properties as incompatible_props

def check_incompatible_props(new, olds):
    if False:
        i = 10
        return i + 15
    '\n    Takes a property and a set of already-seen properties, and checks\n    to see if the new is incompatible with any of the old ones.\n    '
    newly_set = incompatible_props.get(new, set()) | {new}
    for old in olds:
        if newly_set.intersection(incompatible_props.get(old, (old,))):
            return old
    return False

class Parser(object):
    nchildren = 'many'

    def __init__(self, name, child_statement=True):
        if False:
            for i in range(10):
                print('nop')
        self.name = name
        self.positional = []
        self.keyword = {}
        self.children = {}
        statements[name] = self
        self.variable = False
        if child_statement:
            all_child_statements.append(self)
        global parser
        parser = self

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '<%s: %s>' % (self.__class__.__name__, self.name)

    def add(self, i):
        if False:
            i = 10
            return i + 15
        '\n        Adds a clause to this parser.\n        '
        if isinstance(i, list):
            for j in i:
                self.add(j)
            return
        if isinstance(i, Positional):
            self.positional.append(i)
        elif isinstance(i, Keyword):
            self.keyword[i.name] = i
        elif isinstance(i, Style):
            for j in STYLE_PREFIXES:
                self.keyword[j + i.name] = i
        elif isinstance(i, PrefixStyle):
            for j in STYLE_PREFIXES:
                self.keyword[i.prefix + j + i.name] = i
        elif isinstance(i, Parser):
            self.children[i.name] = i

    def parse_statement(self, loc, l, layout_mode=False, keyword=True):
        if False:
            print('Hello World!')
        word = l.word() or l.match('\\$')
        if word and word in self.children:
            if layout_mode:
                c = self.children[word].parse_layout(loc, l, self, keyword)
            else:
                c = self.children[word].parse(loc, l, self, keyword)
            return c
        else:
            return None

    def parse_layout(self, loc, l, parent, keyword):
        if False:
            for i in range(10):
                print('nop')
        l.error('The %s statement cannot be used as a container for the has statement.' % self.name)

    def parse(self, loc, l, parent, keyword):
        if False:
            while True:
                i = 10
        '\n        This is expected to parse a function statement, and to return\n        a list of python ast statements.\n\n        `loc`\n            The location of the current statement.\n\n        `l`\n            The lexer.\n\n        `parent`\n            The parent Parser of the current statement.\n        '
        raise Exception('Not Implemented')

    def parse_contents(self, l, target, layout_mode=False, can_has=False, can_tag=False, block_only=False, keyword=True):
        if False:
            while True:
                i = 10
        "\n        Parses the remainder of the current line of `l`, and all of its subblock,\n        looking for keywords and children.\n\n        `layout_mode`\n            If true, parsing continues to the end of `l`, rather than stopping\n            with the end of the first logical line.\n\n        `can_has`\n            If true, we should parse layouts.\n\n        `can_tag`\n            If true, we should parse the ``tag`` keyword, as it's used by\n            screens.\n\n        `block_only`\n            If true, only parse the block and not the initial properties.\n        "
        seen_keywords = set()
        block = False

        def parse_keyword(l, expect, first_line):
            if False:
                return 10
            name = l.word()
            if name is None:
                l.error(expect)
            if can_tag and name == 'tag':
                if target.tag is not None:
                    l.error('the tag keyword argument appears more than once in a %s statement.' % (self.name,))
                target.tag = l.require(l.word)
                l.expect_noblock(name)
                return True
            if self.variable:
                if name == 'as':
                    if target.variable is not None:
                        l.error('an as clause may only appear once in a %s statement.' % (self.name,))
                    target.variable = l.require(l.word)
                    return
            if name not in self.keyword:
                if name == 'continue' or name == 'break':
                    l.error('The %s statement may only appear inside a for statement, or an if statement inside a for statement.' % name)
                elif name in statements:
                    l.error('The %s statement is not a valid child of the %s statement.' % (name, self.name))
                else:
                    l.error('%r is not a keyword argument or valid child of the %s statement.' % (name, self.name))
            if name == 'at' and l.keyword('transform'):
                if target.atl_transform is not None:
                    l.error("More than one 'at transform' block is given.")
                l.require(':')
                l.expect_eol()
                l.expect_block('ATL block')
                expr = renpy.atl.parse_atl(l.subblock_lexer())
                target.atl_transform = expr
                return
            if name in seen_keywords:
                l.error('keyword argument %r appears more than once in a %s statement.' % (name, self.name))
            incomprop = check_incompatible_props(name, seen_keywords)
            if incomprop:
                l.deferred_error('check_conflicting_properties', 'keyword argument {!r} is incompatible with {!r}.'.format(name, incomprop))
            if name == 'at' and target.atl_transform:
                l.error("The 'at' property must occur before the 'at transform' block.")
            seen_keywords.add(name)
            expr = l.comma_expression()
            if expr is None:
                l.error('the {} keyword argument was not given a value.'.format(name))
            if not keyword and (not renpy.config.keyword_after_python):
                try:
                    literal_eval(expr)
                except Exception:
                    l.error("a non-constant keyword argument like '%s %s' is not allowed after a python block." % (name, expr))
            target.keyword.append((name, expr))
            if not first_line:
                l.expect_noblock(name)
        if block_only:
            l.expect_eol()
            l.expect_block(self.name)
            block = True
        else:
            while True:
                if l.match(':'):
                    l.expect_eol()
                    l.expect_block(self.name)
                    block = True
                    break
                if l.eol():
                    if not target.atl_transform:
                        l.expect_noblock(self.name)
                    block = False
                    break
                parse_keyword(l, 'expected a keyword argument, colon, or end of line.', True)
        lexers = []
        if block:
            lexers.append(l.subblock_lexer())
        if layout_mode:
            lexers.append(l)
        for l in lexers:
            while l.advance():
                state = l.checkpoint()
                loc = l.get_location()
                if l.keyword('has'):
                    if not can_has:
                        l.error('The has statement is not allowed here.')
                    if target.has_noncondition_child():
                        l.error('The has statement may not be given after a child has been supplied.')
                    c = self.parse_statement(loc, l, layout_mode=True, keyword=keyword)
                    if c is None:
                        l.error('Has expects a child statement.')
                    target.children.append(c)
                    if c.has_python():
                        keyword = False
                    continue
                c = self.parse_statement(loc, l)
                if isinstance(c, slast.SLPass):
                    continue
                if c is not None:
                    target.children.append(c)
                    if c.has_python():
                        keyword = False
                    continue
                l.revert(state)
                if not l.eol():
                    parse_keyword(l, 'expected a keyword argument or child statement.', False)
                while not l.eol():
                    parse_keyword(l, 'expected a keyword argument or end of line.', False)

    def add_positional(self, name):
        if False:
            print('Hello World!')
        global parser
        parser = self
        Positional(name)
        return self

    def add_property(self, name):
        if False:
            for i in range(10):
                print('nop')
        global parser
        parser = self
        Keyword(name)
        return self

    def add_style_property(self, name):
        if False:
            print('Hello World!')
        global parser
        parser = self
        Style(name)
        return self

    def add_prefix_style_property(self, prefix, name):
        if False:
            while True:
                i = 10
        global parser
        parser = self
        PrefixStyle(prefix, name)
        return self

    def add_property_group(self, group, prefix=''):
        if False:
            for i in range(10):
                print('nop')
        global parser
        parser = self
        if group not in renpy.sl2.slproperties.property_groups:
            raise Exception('{!r} is not a known property group.'.format(group))
        for prop in renpy.sl2.slproperties.property_groups[group]:
            if isinstance(prop, Keyword):
                Keyword(prefix + prop.name)
            else:
                PrefixStyle(prefix, prop.name)
        return self

    def copy_properties(self, name):
        if False:
            i = 10
            return i + 15
        global parser
        parser = self
        parser_to_copy = statements.get(name, None)
        if parser_to_copy is None:
            raise Exception('{!r} is not a known screen statement'.format(name))
        for p in parser_to_copy.positional:
            Positional(p.name)
        for v in set(parser_to_copy.keyword.values()):
            if isinstance(v, Keyword):
                Keyword(v.name)
            elif isinstance(v, Style):
                Style(v.name)
            elif isinstance(v, PrefixStyle):
                PrefixStyle(v.prefix, v.name)
        return self

def add(thing):
    if False:
        print('Hello World!')
    parser.add(thing)
many = renpy.object.Sentinel('many')

def register_sl_displayable(*args, **kwargs):
    if False:
        while True:
            i = 10
    '\n    :doc: custom_sl class\n    :args: (name, displayable, style, nchildren=0, scope=False, *, replaces=False, default_keywords={}, default_properties=True, unique=False)\n\n    Registers a screen language statement that creates a displayable.\n\n    `name`\n        The name of the screen language statement, a string containing a Ren\'Py\n        keyword. This keyword is used to introduce the new statement.\n\n    `displayable`\n        This is a function that, when called, returns a displayable\n        object. All position arguments, properties, and style properties\n        are passed as arguments to this function. Other keyword arguments\n        are also given to this function, a described below.\n\n        This must return a Displayable. If it returns multiple displayables,\n        the _main attribute of the outermost displayable should be set to\n        the "main" displayable - the one that children should be added\n        to.\n\n    `style`\n        The base name of the style of this displayable. If the style property\n        is not given, this will have the style prefix added to it. The\n        computed style is passed to the `displayable` function as the\n        ``style`` keyword argument.\n\n    `nchildren`\n        The number of children of this displayable. One of:\n\n        0\n            The displayable takes no children.\n        1\n            The displayable takes 1 child. If more than one child is given,\n            the children are placed in a Fixed.\n        "many"\n            The displayable takes more than one child.\n\n\n    `unique`\n        This should be set to true if the function returns a  displayable with\n        no other references to it.\n\n    The following arguments should be passed in using keyword arguments:\n\n    `replaces`\n        If true, and the displayable replaces a prior displayable, that displayable\n        is passed as a parameter to the new displayable.\n\n    `default_keywords`\n        The default set of keyword arguments to supply to the displayable.\n\n    `default_properties`\n        If true, the ui and position properties are added by default.\n\n    Returns an object that can have positional arguments and properties\n    added to it by calling the following methods. Each of these methods\n    returns the object it is called on, allowing methods to be chained\n    together.\n\n    .. method:: add_positional(name)\n\n        Adds a positional argument with `name`\n\n    .. method:: add_property(name)\n\n        Adds a property with `name`. Properties are passed as keyword\n        arguments.\n\n    .. method:: add_style_property(name)\n\n        Adds a family of properties, ending with `name` and prefixed with\n        the various style property prefixes. For example, if called with\n        ("size"), this will define size, idle_size, hover_size, etc.\n\n    .. method:: add_prefix_style_property(prefix, name)\n\n        Adds a family of properties with names consisting of `prefix`,\n        a style property prefix, and `name`. For example, if called\n        with a prefix of `text_` and a name of `size`, this will\n        create text_size, text_idle_size, text_hover_size, etc.\n\n    .. method:: add_property_group(group, prefix=\'\')\n\n        Adds a group of properties, prefixed with `prefix`. `Group` may\n        be one of the strings:\n\n        * "bar"\n        * "box"\n        * "button"\n        * "position"\n        * "text"\n        * "window"\n\n        These correspond to groups of :doc:`style_properties`. Group can\n        also be "ui", in which case it adds the :ref:`common ui properties <common-properties>`.\n    \n    .. method:: copy_properties(name)\n\n        Adds all styles and positional/keyword arguments that can be passed to the `name` screen statement.\n    '
    kwargs.setdefault('unique', False)
    rv = DisplayableParser(*args, **kwargs)
    for i in childbearing_statements:
        i.add(rv)
    screen_parser.add(rv)
    if rv.nchildren != 0:
        childbearing_statements.add(rv)
        for i in all_child_statements:
            rv.add(i)
    rv.add(if_statement)
    rv.add(pass_statement)
    return rv

class DisplayableParser(Parser):

    def __init__(self, name, displayable, style, nchildren=0, scope=False, pass_context=False, imagemap=False, replaces=False, default_keywords={}, hotspot=False, default_properties=True, unique=False):
        if False:
            print('Hello World!')
        '\n        `scope`\n            If true, the scope is passed into the displayable function as a keyword\n            argument named "scope".\n\n        `pass_context`\n            If true, the context is passed as the first positional argument of the\n            displayable.\n\n        `imagemap`\n            If true, the displayable is treated as defining an imagemap. (The imagemap\n            is added to and removed from renpy.ui.imagemap_stack as appropriate.)\n\n        `hotspot`\n            If true, the displayable is treated as a hotspot. (It needs to be\n            re-created if the imagemap it belongs to has changed.)\n\n        `default_properties`\n            If true, the ui and positional properties are added by default.\n        '
        super(DisplayableParser, self).__init__(name)
        self.displayable = displayable
        if nchildren == 'many':
            nchildren = many
        self.nchildren = nchildren
        if nchildren != 0:
            childbearing_statements.add(self)
        self.style = style
        self.scope = scope
        self.pass_context = pass_context
        self.imagemap = imagemap
        self.hotspot = hotspot
        self.replaces = replaces
        self.default_keywords = default_keywords
        self.variable = True
        self.unique = unique
        Keyword('arguments')
        Keyword('properties')
        Keyword('prefer_screen_to_id')
        if default_properties:
            add(renpy.sl2.slproperties.ui_properties)
            add(renpy.sl2.slproperties.position_properties)

    def parse_layout(self, loc, l, parent, keyword):
        if False:
            while True:
                i = 10
        return self.parse(loc, l, parent, keyword, layout_mode=True)

    def parse(self, loc, l, parent, keyword, layout_mode=False):
        if False:
            i = 10
            return i + 15
        rv = slast.SLDisplayable(loc, self.displayable, scope=self.scope, child_or_fixed=self.nchildren == 1, style=self.style, pass_context=self.pass_context, imagemap=self.imagemap, replaces=self.replaces, default_keywords=self.default_keywords, hotspot=self.hotspot, name=self.name, unique=self.unique)
        for _i in self.positional:
            expr = l.simple_expression()
            if expr is None:
                break
            rv.positional.append(expr)
        can_has = self.nchildren == 1
        self.parse_contents(l, rv, layout_mode=layout_mode, can_has=can_has, can_tag=False)
        if len(rv.positional) != len(self.positional):
            if not rv.keyword_exist('arguments'):
                l.error('{} statement expects {} positional arguments, got {}.'.format(self.name, len(self.positional), len(rv.positional)))
        return rv

class IfParser(Parser):

    def __init__(self, name, node_type, parent_contents):
        if False:
            while True:
                i = 10
        '\n        `node_type`\n            The type of node to create.\n\n        `parent_contents`\n            If true, our children must be children of our parent. Otherwise,\n            our children must be children of ourself.\n        '
        super(IfParser, self).__init__(name)
        self.node_type = node_type
        self.parent_contents = parent_contents
        if not parent_contents:
            childbearing_statements.add(self)

    def parse(self, loc, l, parent, keyword):
        if False:
            print('Hello World!')
        if self.parent_contents:
            contents_from = parent
        else:
            contents_from = self
        rv = self.node_type(loc)
        condition = l.require(l.python_expression)
        l.require(':')
        block = slast.SLBlock(loc)
        contents_from.parse_contents(l, block, block_only=True)
        rv.entries.append((condition, block))
        state = l.checkpoint()
        while l.advance():
            loc = l.get_location()
            if l.keyword('elif'):
                condition = l.require(l.python_expression)
                l.require(':')
                block = slast.SLBlock(loc)
                contents_from.parse_contents(l, block, block_only=True, keyword=keyword)
                rv.entries.append((condition, block))
                state = l.checkpoint()
            elif l.keyword('else'):
                condition = None
                l.require(':')
                block = slast.SLBlock(loc)
                contents_from.parse_contents(l, block, block_only=True, keyword=keyword)
                rv.entries.append((condition, block))
                state = l.checkpoint()
                break
            else:
                l.revert(state)
                break
        return rv
if_statement = IfParser('if', slast.SLIf, True)
IfParser('showif', slast.SLShowIf, False)

class ForParser(Parser):

    def __init__(self, name):
        if False:
            i = 10
            return i + 15
        super(ForParser, self).__init__(name)
        childbearing_statements.add(self)

    def name_or_tuple_pattern(self, l):
        if False:
            i = 10
            return i + 15
        '\n        Matches either a name or a tuple pattern. If a single name is being\n        matched, returns it. Otherwise, returns None.\n        '
        name = None
        pattern = False
        while True:
            if l.match('\\('):
                name = self.name_or_tuple_pattern(l)
                l.require('\\)')
                pattern = True
            else:
                name = l.name()
                if name is None:
                    break
            if l.match(','):
                pattern = True
            else:
                break
        if pattern:
            return None
        if name is not None:
            return name
        l.error('expected variable or tuple pattern.')

    def parse(self, loc, l, parent, keyword):
        if False:
            i = 10
            return i + 15
        l.skip_whitespace()
        tuple_start = l.pos
        name = self.name_or_tuple_pattern(l)
        if not name:
            name = '_sl2_i'
            pattern = l.text[tuple_start:l.pos]
            stmt = pattern + ' = ' + name
            code = renpy.ast.PyCode(stmt, loc)
        else:
            code = None
        if l.match('index'):
            index_expression = l.require(l.say_expression)
        else:
            index_expression = None
        l.require('in')
        expression = l.require(l.python_expression)
        l.require(':')
        l.expect_eol()
        rv = slast.SLFor(loc, name, expression, index_expression)
        if code:
            rv.children.append(slast.SLPython(loc, code))
        self.parse_contents(l, rv, block_only=True)
        return rv
for_parser = ForParser('for')

class BreakParser(Parser):

    def parse(self, loc, l, parent, keyword):
        if False:
            return 10
        l.expect_eol()
        l.expect_noblock('break statement')
        return slast.SLBreak(loc)
for_parser.add(BreakParser('break', False))

class ContinueParser(Parser):

    def parse(self, loc, l, parent, keyword):
        if False:
            for i in range(10):
                print('nop')
        l.expect_eol()
        l.expect_noblock('continue statement')
        return slast.SLContinue(loc)
for_parser.add(ContinueParser('continue', False))

class OneLinePythonParser(Parser):

    def parse(self, loc, l, parent, keyword):
        if False:
            return 10
        loc = l.get_location()
        source = l.require(l.rest_statement)
        l.expect_eol()
        l.expect_noblock('one-line python')
        code = renpy.ast.PyCode(source, loc)
        return slast.SLPython(loc, code)
OneLinePythonParser('$')

class MultiLinePythonParser(Parser):

    def parse(self, loc, l, parent, keyword):
        if False:
            while True:
                i = 10
        loc = l.get_location()
        l.require(':')
        l.expect_eol()
        l.expect_block('python block')
        source = l.python_block()
        code = renpy.ast.PyCode(source, loc)
        return slast.SLPython(loc, code)
MultiLinePythonParser('python')

class PassParser(Parser):

    def parse(self, loc, l, parent, keyword):
        if False:
            i = 10
            return i + 15
        l.expect_eol()
        l.expect_noblock('pass statement')
        return slast.SLPass(loc)
pass_statement = PassParser('pass')

class DefaultParser(Parser):

    def parse(self, loc, l, parent, keyword):
        if False:
            i = 10
            return i + 15
        name = l.require(l.word)
        l.require('=')
        rest = l.rest()
        l.expect_eol()
        l.expect_noblock('default statement')
        return slast.SLDefault(loc, name, rest)
DefaultParser('default')

class UseParser(Parser):

    def __init__(self, name):
        if False:
            for i in range(10):
                print('nop')
        super(UseParser, self).__init__(name)
        childbearing_statements.add(self)

    def parse(self, loc, l, parent, keyword):
        if False:
            return 10
        if l.keyword('expression'):
            target = l.require(l.simple_expression)
            l.keyword('pass')
        else:
            target = l.require(l.word)
        args = renpy.parser.parse_arguments(l)
        if l.keyword('id'):
            id_expr = l.simple_expression()
        else:
            id_expr = None
        if l.match(':'):
            l.expect_eol()
            l.expect_block('use statement')
            block = slast.SLBlock(loc)
            self.parse_contents(l, block, can_has=True, block_only=True)
        else:
            l.expect_eol()
            l.expect_noblock('use statement')
            block = None
        return slast.SLUse(loc, target, args, id_expr, block)
UseParser('use')
Keyword('style_prefix')
Keyword('style_group')

class TranscludeParser(Parser):

    def parse(self, loc, l, parent, keyword):
        if False:
            for i in range(10):
                print('nop')
        l.expect_eol()
        return slast.SLTransclude(loc)
TranscludeParser('transclude')

class CustomParser(Parser):
    """
    :doc: custom_sl class
    :name: renpy.register_sl_statement

    Registers a custom screen language statement with Ren'Py.

    `name`
        This must be a word. It's the name of the custom screen language
        statement.

    `children`
        The number of children this custom statement takes. This should
        be 0, 1, or "many", which means zero or more.

    `screen`
        The screen to use. If not given, defaults to `name`.

    Returns an object that can have positional arguments and properties
    added to it. This object has the same .add_ methods as the objects
    returned by :class:`renpy.register_sl_displayable`.
    """

    def __init__(self, name, children='many', screen=None):
        if False:
            while True:
                i = 10
        Parser.__init__(self, name)
        if children == 'many':
            children = many
        for i in childbearing_statements:
            i.add(self)
        screen_parser.add(self)
        self.nchildren = children
        if self.nchildren != 0:
            childbearing_statements.add(self)
            for i in all_child_statements:
                self.add(i)
        self.add_property('arguments')
        self.add_property('properties')
        self.add(if_statement)
        self.add(pass_statement)
        global parser
        parser = None
        if screen is not None:
            self.screen = screen
        else:
            self.screen = name

    def parse(self, loc, l, parent, keyword):
        if False:
            while True:
                i = 10
        arguments = []
        for _i in self.positional:
            expr = l.simple_expression()
            if expr is None:
                break
            arguments.append(expr)
        block = slast.SLBlock(loc)
        can_has = self.nchildren == 1
        self.parse_contents(l, block, can_has=can_has, can_tag=False)
        if len(arguments) != len(self.positional):
            if not block.keyword_exist('arguments'):
                l.error('{} statement expects {} positional arguments, got {}.'.format(self.name, len(self.positional), len(arguments)))
        return slast.SLCustomUse(loc, self.screen, arguments, block)

class ScreenParser(Parser):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super(ScreenParser, self).__init__('screen', child_statement=False)

    def parse(self, loc, l, parent, name='_name', keyword=True):
        if False:
            for i in range(10):
                print('nop')
        screen = slast.SLScreen(loc)
        screen.name = l.require(l.word)
        screen.parameters = renpy.parser.parse_parameters(l)
        self.parse_contents(l, screen, can_tag=True)
        keyword = dict(screen.keyword)
        screen.modal = keyword.get('modal', 'False')
        screen.zorder = keyword.get('zorder', '0')
        screen.variant = keyword.get('variant', 'None')
        screen.predict = keyword.get('predict', 'None')
        screen.layer = keyword.get('layer', "'screens'")
        screen.sensitive = keyword.get('sensitive', 'True')
        screen.roll_forward = keyword.get('roll_forward', 'None')
        return screen
screen_parser = ScreenParser()
Keyword('modal')
Keyword('zorder')
Keyword('variant')
Keyword('predict')
Keyword('style_group')
Keyword('style_prefix')
Keyword('layer')
Keyword('sensitive')
Keyword('roll_forward')
parser = None

def init():
    if False:
        i = 10
        return i + 15
    screen_parser.add(all_child_statements)
    for i in all_child_statements:
        if i in childbearing_statements:
            i.add(all_child_statements)
        else:
            i.add(if_statement)
            i.add(pass_statement)

def parse_screen(l, loc):
    if False:
        return 10
    '\n    Parses the screen statement.\n    '
    return screen_parser.parse(loc, l, None)