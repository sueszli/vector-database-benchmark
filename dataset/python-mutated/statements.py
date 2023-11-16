from __future__ import division, absolute_import, with_statement, print_function, unicode_literals
from renpy.compat import PY2, basestring, bchr, bord, chr, open, pystr, range, round, str, tobytes, unicode
import renpy
registry = {}
parsers = renpy.parser.ParseTrie()

def register(name, parse=None, lint=None, execute=None, predict=None, next=None, scry=None, block=False, init=False, translatable=False, execute_init=None, init_priority=0, label=None, warp=None, translation_strings=None, force_begin_rollback=False, post_execute=None, post_label=None, predict_all=True, predict_next=None, execute_default=None, reachable=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    :doc: statement_register\n    :name: renpy.register_statement\n\n    This registers a user-defined statement.\n\n    `name`\n        This is either a space-separated list of names that begin the statement, or the\n        empty string to define a new default statement (the default statement will\n        replace the say statement).\n\n    `block`\n        When this is False, the statement does not expect a block. When True, it\n        expects a block, but leaves it up to the lexer to parse that block. If the\n        string "script", the block is interpreted as containing one or more\n        Ren\'Py script language statements. If the string "possible", the\n        block expect condition is determined by the parse function.\n\n    `parse`\n        This is a function that takes a Lexer object. This function should parse the\n        statement, and return an object. This object is passed as an argument to all the\n        other functions.\n\n    `lint`\n        This is called to check the statement. It is passed a single argument, the\n        object returned from parse. It should call renpy.error to report errors.\n\n    `execute`\n        This is a function that is called when the statement executes. It is passed a\n        single argument, the object returned from parse.\n\n    `execute_init`\n        This is a function that is called at init time, at priority 0. It is passed a\n        single argument, the object returned from parse.\n\n    `predict`\n        This is a function that is called to predict the images used by the statement.\n        It is passed a single argument, the object returned from parse. It should return\n        a list of displayables used by the statement.\n\n    `next`\n        This is a function that is called to determine the next statement.\n\n        If `block` is not "script", this is passed a single argument, the object\n        returned from the parse function. If `block` is "script", an additional\n        argument is passed, an object that names the first statement in the block.\n\n        The function should return either a string giving a label to jump to,\n        the second argument to transfer control into the block, or None to\n        continue to the statement after this one. It can also return the result\n        of :meth:`Lexer.renpy_statement` or :meth:`Lexer.renpy_block` when\n        called in the `parse` function.\n\n    `label`\n        This is a function that is called to determine the label of this\n        statement. If it returns a string, that string is used as the statement\n        label, which can be called and jumped to like any other label.\n\n    `warp`\n        This is a function that is called to determine if this statement\n        should execute during warping. If the function exists and returns\n        true, it\'s run during warp, otherwise the statement is not run\n        during warp.\n\n    `scry`\n        Used internally by Ren\'Py.\n\n    `init`\n        True if this statement should be run at init-time. (If the statement\n        is not already inside an init block, it\'s automatically placed inside\n        an init block.)\n\n        You probably don\'t want this if you have an `execute_init` function,\n        as wrapping the statement in an init block will cause the `execute_init`\n        and `execute` functions to be called at the same time.\n\n    `init_priority`\n        An integer that determines the priority of initialization of the\n        init block created by `init` and `execute_init` function.\n\n    `translation_strings`\n        A function that is called with the parsed block. It\'s expected to\n        return a list of strings, which are then reported as being available\n        to be translated.\n\n    `force_begin_rollback`\n        This should be set to true on statements that are likely to cause the\n        end of a fast skip, similar to ``menu`` or ``call screen``.\n\n    `post_execute`\n        A function that is executed as part the next statement after this\n        one. (Adding a post_execute function changes the contents of the RPYC\n        file, meaning a Force Compile is necessary.)\n\n    `post_label`\n        This is a function that is called to determine the label of this\n        the post execute statement. If it returns a string, that string is used\n        as the statement label, which can be called and jumped to like any other\n        label. This can be used to create a unique return point.\n\n    `predict_all`\n        If True, then this predicts all sub-parses of this statement and\n        the statement after this statement.\n\n    `predict_next`\n        This is called with a single argument, the label of the statement\n        that would run after this statement.\n\n        This should be called to predict the statements that can run after\n        this one. It\'s expected to return a list of of labels or SubParse\n        objects. This is not called if `predict_all` is true.\n\n    `execute_default`\n        This is a function that is called at the same time the default\n        statements are run - after the init phase, but before the game starts; when the\n        a save is loaded; after rollback; before lint; and potentially at\n        other times.\n\n        This is called with a single argument, the object returned from parse.\n\n    `reachable`\n        This is a function that is called to allow this statement to\n        customize how it participates in lint\'s reachability analysis.\n\n        By default, a statement\'s custom block, sub-parse blocks created\n        with Lexer.renpy_block(), and the statement after the statement\n        are reachable if the statement itself is reachable. The statement\n        is also reachable if it has a label function.\n\n        This can be customized by providing a reachable function. This is\n        a function that takes five arguments (in the following, a "label"\n        may be a string or an opaque object):\n\n        * The object returned by the parse function.\n        * A boolean that is true if the statement is reachable.\n        * The label of the statement.\n        * The label of the next statement, or None if there is no next statement.\n        * If `block` is set to "script", the label of the first statement in the block,\n          or None if there is no block.\n\n        It\'s expected to return a set that may contain:\n\n        * A label or subparse object of a statement that is reachable.\n        * True, to indicate that this statement should not be reported by lint,\n          but is not intrinsically reachable. (It will become reachable if it\n          is reported reachable by another statement.)\n        * None, which is ignored.\n\n        This function may be called multiple times with both value of is_reachable,\n        to allow the statement to customize its behavior based on whether it\'s\n        reachable or not. (For example, the next statement may only be reachable\n        if this statement is.)\n\n    .. warning::\n\n        Using the empty string as the name to redefine the say statement is\n        usually a bad idea. That is because when replacing a Ren\'Py native\n        statement, its behavior depends on the :doc:`statement_equivalents`. In\n        the case of the say statement, these equivalents do not support the `id`\n        and translation systems. As a result, a game redefining the default\n        statement will not be able to use these features (short of\n        reimplementing them entirely).\n    '
    name = tuple(name.split())
    if label:
        force_begin_rollback = True
    registry[name] = dict(parse=parse, lint=lint, execute=execute, execute_init=execute_init, predict=predict, next=next, scry=scry, label=label, warp=warp, translation_strings=translation_strings, rollback='force' if force_begin_rollback else 'normal', post_execute=post_execute, post_label=post_label, predict_all=predict_all, predict_next=predict_next, execute_default=execute_default, reachable=reachable)
    if block not in [True, False, 'script', 'possible']:
        raise Exception('Unknown "block" argument value: {}'.format(block))

    def parse_user_statement(l, loc):
        if False:
            i = 10
            return i + 15
        renpy.exports.push_error_handler(l.error)
        old_subparses = l.subparses
        try:
            l.subparses = []
            text = l.text
            subblock = l.subblock
            code_block = None
            if block is False:
                l.expect_noblock(' '.join(name) + ' statement')
            elif block is True:
                l.expect_block(' '.join(name) + ' statement')
            elif block == 'script':
                l.expect_block(' '.join(name) + ' statement')
                code_block = renpy.parser.parse_block(l.subblock_lexer())
            start_line = l.line
            parsed = (name, parse(l))
            if l.line == start_line:
                l.advance()
            rv = renpy.ast.UserStatement(loc, text, subblock, parsed)
            rv.translatable = translatable
            rv.translation_relevant = bool(translation_strings)
            rv.code_block = code_block
            rv.subparses = l.subparses
            rv.init_priority = init_priority + l.init_offset
        finally:
            l.subparses = old_subparses
            renpy.exports.pop_error_handler()
        if post_execute is not None or post_label is not None:
            post = renpy.ast.PostUserStatement(loc, rv)
            rv = [rv, post]
        if init and (not l.init):
            rv = renpy.ast.Init(loc, [rv], init_priority + l.init_offset)
        return rv
    renpy.parser.statements.add(name, parse_user_statement)

    def parse_data(l):
        if False:
            for i in range(10):
                print('nop')
        return (name, registry[name]['parse'](l))
    parsers.add(name, parse_data)

def parse(node, line, subblock):
    if False:
        while True:
            i = 10
    '\n    This is used for runtime parsing of CDSes that were created before 7.3.\n    '
    block = [(node.filename, node.linenumber, line, subblock)]
    l = renpy.parser.Lexer(block)
    l.advance()
    renpy.exports.push_error_handler(l.error)
    try:
        pf = parsers.parse(l)
        if pf is None:
            l.error('Could not find user-defined statement at runtime.')
        return pf(l)
    finally:
        renpy.exports.pop_error_handler()

def call(method, parsed, *args, **kwargs):
    if False:
        print('Hello World!')
    (name, parsed) = parsed
    method = registry[name].get(method)
    if method is None:
        return None
    return method(parsed, *args, **kwargs)

def get(key, parsed):
    if False:
        for i in range(10):
            print('nop')
    (name, parsed) = parsed
    return registry[name].get(key, None)

def get_name(parsed):
    if False:
        i = 10
        return i + 15
    (name, _parsed) = parsed
    return ' '.join(name)