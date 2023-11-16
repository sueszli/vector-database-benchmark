from __future__ import division, absolute_import, with_statement, print_function, unicode_literals
from renpy.compat import PY2, basestring, bchr, bord, chr, open, pystr, range, round, str, tobytes, unicode
import renpy
import re
import codecs
lines = {}
files = set()

class Line(object):
    """
    Represents a logical line in a file.
    """

    def __init__(self, filename, number, start):
        if False:
            print('Hello World!')
        filename = filename.replace('\\', '/')
        self.filename = filename
        self.number = number
        self.start = start
        self.end = start
        self.end_delim = start
        self.text = ''
        self.full_text = ''

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '<Line {}:{} {!r}>'.format(self.filename, self.number, self.text)

def ensure_loaded(filename):
    if False:
        for i in range(10):
            print('nop')
    "\n    Ensures that the given filename and linenumber are loaded. Doesn't do\n    anything if the filename can't be loaded.\n    "
    if not (filename.endswith('.rpy') or filename.endswith('.rpyc')):
        return
    if filename in files:
        return
    files.add(filename)
    fn = renpy.lexer.unelide_filename(filename)
    renpy.lexer.list_logical_lines(fn, add_lines=True)

def get_line_text(filename, linenumber):
    if False:
        while True:
            i = 10
    '\n    Gets the text of the line with `filename` and `linenumber`, or the None if\n    the line does not exist.\n    '
    filename = filename.replace('\\', '/')
    ensure_loaded(filename)
    if (filename, linenumber) in lines:
        return lines[filename, linenumber].text
    else:
        return None

def adjust_line_locations(filename, linenumber, char_offset, line_offset):
    if False:
        return 10
    '\n    Adjusts the locations in the line data structure.\n\n    `filename`, `linenumber`\n        The filename and first line number to adjust.\n\n    `char_offset`\n        The number of characters in the file to offset the code by,.\n\n    `line_offset`\n        The number of line in the file to offset the code by.\n    '
    filename = filename.replace('\\', '/')
    ensure_loaded(filename)
    global lines
    new_lines = {}
    for (key, line) in lines.items():
        (fn, ln) = key
        if fn == filename and linenumber <= ln:
            ln += line_offset
            line.number += line_offset
            line.start += char_offset
            line.end += char_offset
            line.end_delim += char_offset
        new_lines[fn, ln] = line
    lines = new_lines

def insert_line_before(code, filename, linenumber):
    if False:
        print('Hello World!')
    '\n    Adds `code` immediately before `filename` and `linenumber`. Those must\n    correspond to an existing line, and the code is inserted with the same\n    indentation as that line.\n    '
    filename = filename.replace('\\', '/')
    if renpy.config.clear_lines:
        raise Exception('config.clear_lines must be False for script editing to work.')
    ensure_loaded(filename)
    old_line = lines[filename, linenumber]
    m = re.match(' *', old_line.text)
    indent = m.group(0)
    if not code:
        indent = ''
    if old_line.text.endswith('\r\n') or not old_line.text.endswith('\n'):
        line_ending = '\r\n'
    else:
        line_ending = '\n'
    raw_code = indent + code
    code = indent + code + line_ending
    new_line = Line(old_line.filename, old_line.number, old_line.start)
    new_line.text = raw_code
    new_line.full_text = code
    new_line.end = new_line.start + len(raw_code)
    new_line.end_delim = new_line.start + len(code)
    with codecs.open(old_line.filename, 'r', 'utf-8') as f:
        data = f.read()
    data = data[:old_line.start] + code + data[old_line.start:]
    adjust_line_locations(filename, linenumber, len(code), code.count('\n'))
    with renpy.loader.auto_lock:
        with codecs.open(old_line.filename, 'w', 'utf-8') as f:
            f.write(data)
        renpy.loader.add_auto(old_line.filename, force=True)
    lines[filename, linenumber] = new_line

def remove_line(filename, linenumber):
    if False:
        return 10
    '\n    Removes `linenumber` from `filename`. The line must exist and correspond\n    to a logical line.\n    '
    filename = filename.replace('\\', '/')
    if renpy.config.clear_lines:
        raise Exception('config.clear_lines must be False for script editing to work.')
    ensure_loaded(filename)
    line = lines[filename, linenumber]
    with codecs.open(line.filename, 'r', 'utf-8') as f:
        data = f.read()
    code = data[line.start:line.end_delim]
    data = data[:line.start] + data[line.end_delim:]
    del lines[filename, linenumber]
    adjust_line_locations(filename, linenumber, -len(code), -code.count('\n'))
    with renpy.loader.auto_lock:
        with codecs.open(line.filename, 'w', 'utf-8') as f:
            f.write(data)
        renpy.loader.add_auto(line.filename, force=True)

def get_full_text(filename, linenumber):
    if False:
        i = 10
        return i + 15
    '\n    Returns the full text of `linenumber` from `filename`, including\n    any comment or delimiter characters that exist.\n    '
    filename = filename.replace('\\', '/')
    ensure_loaded(filename)
    if (filename, linenumber) not in lines:
        return None
    return lines[filename, linenumber].full_text

def nodes_on_line(filename, linenumber):
    if False:
        i = 10
        return i + 15
    '\n    Returns a list of nodes that are found on the given line.\n    '
    ensure_loaded(filename)
    rv = []
    for i in renpy.game.script.all_stmts:
        if i.filename == filename and i.linenumber == linenumber and (i.rollback != 'never'):
            rv.append(i)
    return rv

def nodes_on_line_at_or_after(filename, linenumber):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns a list of nodes that are found at or after the given line.\n    '
    ensure_loaded(filename)
    lines = [i.linenumber for i in renpy.game.script.all_stmts if i.filename == filename if i.linenumber >= linenumber if i.rollback != 'never']
    if not lines:
        return []
    return nodes_on_line(filename, min(lines))

def first_and_last_nodes(nodes):
    if False:
        i = 10
        return i + 15
    '\n    Finds the first and last nodes in `nodes`, a list of nodes. This assumes\n    that all the nodes are "simple", with no control flow, and that all of\n    the relevant nodes are in `nodes`.\n    '
    firsts = []
    lasts = []
    for i in nodes:
        for j in nodes:
            if j.next is i:
                break
        else:
            firsts.append(i)
        for j in nodes:
            if i.next is j:
                break
        else:
            lasts.append(i)
    if len(firsts) != 1:
        raise Exception('Could not find unique first AST node.')
    if len(lasts) != 1:
        raise Exception('Could not find unique last AST node.')
    return (firsts[0], lasts[0])

def adjust_ast_linenumbers(filename, linenumber, offset):
    if False:
        i = 10
        return i + 15
    '\n    This adjusts the line numbers in the the ast.\n\n    `filename`\n        The filename to adjust.\n\n    `linenumber`\n        The first line to adjust.\n\n    `offset`\n        The amount to adjust by. Positive numbers increase the line\n    '
    for i in renpy.game.script.all_stmts:
        if i.filename == filename and i.linenumber >= linenumber:
            i.linenumber += offset

def add_to_ast_before(code, filename, linenumber):
    if False:
        return 10
    "\n    Adds `code`, which must be a textual line of Ren'Py code,\n    before the given filename and line number.\n    "
    nodes = nodes_on_line_at_or_after(filename, linenumber)
    (old, _) = first_and_last_nodes(nodes)
    adjust_ast_linenumbers(old.filename, linenumber, 1)
    (block, _init) = renpy.game.script.load_string(old.filename, code, linenumber=linenumber)
    ret_stmt = block.pop()
    renpy.game.script.all_stmts.remove(ret_stmt)
    if not block:
        return
    for i in renpy.game.script.all_stmts:
        i.replace_next(old, block[0])
    renpy.ast.chain_block(block, old)
    for i in renpy.game.contexts:
        i.replace_node(old, block[0])
    renpy.game.log.replace_node(old, block[0])

def can_add_before(filename, linenumber):
    if False:
        i = 10
        return i + 15
    "\n    Returns True if it's possible to add a line before the given filename\n    and linenumber, and False if it's not possible.\n    "
    try:
        nodes = nodes_on_line(filename, linenumber)
        first_and_last_nodes(nodes)
        return True
    except Exception:
        return False

def remove_from_ast(filename, linenumber):
    if False:
        while True:
            i = 10
    '\n    Removes from the AST all statements that happen to be at `filename`\n    and `linenumber`, then adjusts the line numbers appropriately.\n\n    There\'s an assumption that the statement(s) on the line are "simple",\n    not involving control flow.\n    '
    nodes = nodes_on_line(filename, linenumber)
    (first, last) = first_and_last_nodes(nodes)
    new_stmts = []
    for i in renpy.game.script.all_stmts:
        if i in nodes:
            continue
        i.replace_next(first, last.next)
        new_stmts.append(i)
    renpy.game.script.all_stmts = new_stmts
    namemap = renpy.game.script.namemap
    for k in list(namemap):
        if namemap[k] in nodes:
            namemap[k] = last.next
    adjust_ast_linenumbers(filename, linenumber, -1)
serial = 1

def test_add():
    if False:
        for i in range(10):
            print('nop')
    global serial
    s = "'Hello world %f'" % serial
    serial += 1
    node = renpy.game.script.lookup(renpy.game.context().current)
    filename = node.filename
    linenumber = node.linenumber
    add_to_ast_before(s, filename, linenumber)
    insert_line_before(s, filename, linenumber)
    renpy.exports.restart_interaction()

def test_remove():
    if False:
        while True:
            i = 10
    node = renpy.game.script.lookup(renpy.game.context().current)
    filename = node.filename
    linenumber = node.linenumber
    remove_from_ast(filename, linenumber)
    remove_line(filename, linenumber)
    renpy.exports.rollback(checkpoints=0, force=True, greedy=True)