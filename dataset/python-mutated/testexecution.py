from __future__ import division, absolute_import, with_statement, print_function, unicode_literals
from renpy.compat import PY2, basestring, bchr, bord, chr, open, pystr, range, round, str, tobytes, unicode
import pygame_sdl2
import renpy
testcases = {}
node = None
node_loc = None
state = None
old_state = None
old_loc = None
last_state_change = 0
start_time = None
action = None
labels = set()

def take_name(name):
    if False:
        for i in range(10):
            print('nop')
    '\n    Takes the name of a statement that is about to run.\n    '
    if node is None:
        return
    if isinstance(name, basestring):
        labels.add(name)

class TestJump(Exception):
    """
    An exception that is raised in order to jump to `node`.
    """

    def __init__(self, node):
        if False:
            return 10
        self.node = node

def lookup(name, from_node):
    if False:
        i = 10
        return i + 15
    '\n    Tries to look up the name with `target`. If found, returns it, otherwise\n    raises an exception.\n    '
    if name in testcases:
        return testcases[name]
    raise Exception('Testcase {} not found at {}:{}.'.format(name, from_node.filename, from_node.linenumber))

def execute_node(now, node, state, start):
    if False:
        print('Hello World!')
    '\n    Performs one execution cycle of a node.\n    '
    while True:
        try:
            if state is None:
                state = node.start()
                start = now
            if state is None:
                break
            state = node.execute(state, now - start)
            break
        except TestJump as e:
            node = e.node
            state = None
    if state is None:
        node = None
    return (node, state, start)

def execute():
    if False:
        print('Hello World!')
    '\n    Called periodically by the test code to generate events, if desired.\n    '
    global node
    global state
    global start_time
    global action
    global old_state
    global old_loc
    global last_state_change
    _test = renpy.test.testast._test
    if node is None:
        return
    if renpy.display.interface.suppress_underlay and (not _test.force):
        return
    if _test.maximum_framerate:
        renpy.exports.maximum_framerate(10.0)
    else:
        renpy.exports.maximum_framerate(None)
    for e in pygame_sdl2.event.copy_event_queue():
        if getattr(e, 'test', False):
            return
    if action:
        old_action = action
        action = None
        renpy.display.behavior.run(old_action)
    now = renpy.display.core.get_time()
    (node, state, start_time) = execute_node(now, node, state, start_time)
    labels.clear()
    if node is None:
        renpy.test.testmouse.reset()
        return
    loc = renpy.exports.get_filename_line()
    if old_state != state or old_loc != loc:
        last_state_change = now
    old_state = state
    old_loc = loc
    if now - last_state_change > _test.timeout:
        raise Exception('Testcase stuck at {}:{}.'.format(node_loc[0], node_loc[1]))

def test_command():
    if False:
        return 10
    '\n    The dialogue command. This updates dialogue.txt, a file giving all the dialogue\n    in the game.\n    '
    ap = renpy.arguments.ArgumentParser(description='Runs a testcase.')
    ap.add_argument('testcase', help='The name of a testcase to run.', nargs='?', default='default')
    args = ap.parse_args()
    if args.testcase not in testcases:
        raise Exception('Testcase {} was not found.'.format(args.testcase))
    global node
    node = testcases[args.testcase]
    return True
renpy.arguments.register_command('test', test_command)