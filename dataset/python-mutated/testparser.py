from __future__ import division, absolute_import, with_statement, print_function, unicode_literals
from renpy.compat import PY2, basestring, bchr, bord, chr, open, pystr, range, round, str, tobytes, unicode
import renpy.test.testast as testast
import renpy

def parse_click(l, loc, target):
    if False:
        print('Hello World!')
    rv = testast.Click(loc, target)
    while True:
        if l.keyword('button'):
            rv.button = int(l.require(l.integer))
        elif l.keyword('pos'):
            rv.position = l.require(l.simple_expression)
        elif l.keyword('always'):
            rv.always = True
        else:
            break
    return rv

def parse_type(l, loc, keys):
    if False:
        i = 10
        return i + 15
    rv = testast.Type(loc, keys)
    while True:
        if l.keyword('pattern'):
            rv.pattern = l.require(l.string)
        elif l.keyword('pos'):
            rv.position = l.require(l.simple_expression)
        else:
            break
    return rv

def parse_move(l, loc):
    if False:
        print('Hello World!')
    rv = testast.Move(loc)
    rv.position = l.require(l.simple_expression)
    while True:
        if l.keyword('pattern'):
            rv.pattern = l.require(l.string)
        else:
            break
    return rv

def parse_drag(l, loc):
    if False:
        return 10
    points = l.require(l.simple_expression)
    rv = testast.Drag(loc, points)
    while True:
        if l.keyword('button'):
            rv.button = int(l.require(l.integer))
        elif l.keyword('pattern'):
            rv.pattern = l.require(l.string)
        elif l.keyword('steps'):
            rv.steps = int(l.require(l.integer))
        else:
            break
    return rv

def parse_clause(l, loc):
    if False:
        return 10
    if l.keyword('run'):
        expr = l.require(l.simple_expression)
        return testast.Action(loc, expr)
    elif l.keyword('pause'):
        expr = l.require(l.simple_expression)
        return testast.Pause(loc, expr)
    elif l.keyword('label'):
        name = l.require(l.name)
        return testast.Label(loc, name)
    elif l.keyword('type'):
        name = l.name()
        if name is not None:
            return parse_type(l, loc, [name])
        string = l.require(l.string)
        return parse_type(l, loc, list(string))
    elif l.keyword('drag'):
        return parse_drag(l, loc)
    elif l.keyword('move'):
        return parse_move(l, loc)
    elif l.keyword('click'):
        return parse_click(l, loc, None)
    elif l.keyword('scroll'):
        pattern = l.require(l.string)
        return testast.Scroll(loc, pattern)
    else:
        target = l.string()
        if target:
            return parse_click(l, loc, target)
    l.error('Expected a test language statement or clause.')
    return testast.Click(loc, target)

def parse_statement(l, loc):
    if False:
        print('Hello World!')
    if l.keyword('python'):
        l.require(':')
        l.expect_block('python block')
        source = l.python_block()
        code = renpy.ast.PyCode(source, loc)
        return testast.Python(loc, code)
    if l.keyword('if'):
        l.expect_block('if block')
        condition = parse_clause(l, loc)
        l.require(':')
        block = parse_block(l.subblock_lexer(False), loc)
        return testast.If(loc, condition, block)
    l.expect_noblock('statement')
    if l.match('\\$'):
        source = l.require(l.rest)
        code = renpy.ast.PyCode(source, loc)
        return testast.Python(loc, code)
    elif l.keyword('assert'):
        source = l.require(l.rest)
        return testast.Assert(loc, source)
    elif l.keyword('jump'):
        target = l.require(l.name)
        return testast.Jump(loc, target)
    elif l.keyword('call'):
        target = l.require(l.name)
        return testast.Call(loc, target)
    rv = parse_clause(l, loc)
    if l.keyword('until'):
        right = parse_clause(l, loc)
        rv = testast.Until(loc, rv, right)
    return rv

def parse_block(l, loc):
    if False:
        for i in range(10):
            print('nop')
    '\n    Parses a named block of testcase statements.\n    '
    block = []
    while l.advance():
        stmt = parse_statement(l, l.get_location())
        block.append(stmt)
        l.expect_eol()
    return testast.Block(loc, block)