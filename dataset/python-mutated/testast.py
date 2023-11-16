from __future__ import division, absolute_import, with_statement, print_function, unicode_literals
from renpy.compat import PY2, basestring, bchr, bord, chr, open, pystr, range, round, str, tobytes, unicode
import renpy
from renpy.test.testmouse import click_mouse, move_mouse

class TestSettings(renpy.object.Object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.maximum_framerate = True
        self.timeout = 5.0
        self.force = False
        self.transition_timeout = 5.0
_test = TestSettings()

class Node(object):
    """
    An AST node for a test script.
    """

    def __init__(self, loc):
        if False:
            i = 10
            return i + 15
        (self.filename, self.linenumber) = loc

    def start(self):
        if False:
            i = 10
            return i + 15
        '\n        Called once when the node starts execution.\n\n        This is expected to return a state, or None to advance to the next\n        node.\n        '

    def execute(self, state, t):
        if False:
            while True:
                i = 10
        '\n        Called once each time the screen is drawn.\n\n        `state`\n            The last state that was returned from this node.\n\n        `t`\n            The time since start was called.\n        '
        return state

    def ready(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns True if this node is ready to execute, or False otherwise.\n        '
        return True

    def report(self):
        if False:
            i = 10
            return i + 15
        '\n        Reports the location of this statement. This should only be called\n        in the execute method of leaf nodes of the test tree.\n        '
        renpy.test.testexecution.node_loc = (self.filename, self.linenumber)

class Pattern(Node):
    position = None
    always = False

    def __init__(self, loc, pattern=None):
        if False:
            i = 10
            return i + 15
        Node.__init__(self, loc)
        self.pattern = pattern

    def start(self):
        if False:
            return 10
        return True

    def execute(self, state, t):
        if False:
            i = 10
            return i + 15
        self.report()
        if renpy.display.interface.trans_pause and t < _test.transition_timeout:
            return state
        if self.position is not None:
            position = renpy.python.py_eval(self.position)
        else:
            position = (None, None)
        f = renpy.test.testfocus.find_focus(self.pattern)
        if f is None:
            (x, y) = (None, None)
        else:
            (x, y) = renpy.test.testfocus.find_position(f, position)
        if x is None:
            if self.pattern:
                return state
            else:
                (x, y) = renpy.exports.get_mouse_pos()
        return self.perform(x, y, state, t)

    def ready(self):
        if False:
            print('Hello World!')
        if self.always:
            return True
        f = renpy.test.testfocus.find_focus(self.pattern)
        if f is not None:
            return True
        else:
            return False

    def perform(self, x, y, state, t):
        if False:
            while True:
                i = 10
        return None

class Click(Pattern):
    button = 1

    def perform(self, x, y, state, t):
        if False:
            for i in range(10):
                print('nop')
        click_mouse(self.button, x, y)
        return None

class Move(Pattern):

    def perform(self, x, y, state, t):
        if False:
            i = 10
            return i + 15
        move_mouse(x, y)
        return None

class Scroll(Node):

    def __init__(self, loc, pattern=None):
        if False:
            return 10
        Node.__init__(self, loc)
        self.pattern = pattern

    def start(self):
        if False:
            i = 10
            return i + 15
        return True

    def execute(self, state, t):
        if False:
            i = 10
            return i + 15
        self.report()
        f = renpy.test.testfocus.find_focus(self.pattern)
        if f is None:
            return True
        if not isinstance(f.widget, renpy.display.behavior.Bar):
            return True
        adj = f.widget.adjustment
        if adj.value == adj.range:
            new = 0
        else:
            new = adj.value + adj.page
            if new > adj.range:
                new = adj.range
        adj.change(new)
        return None

    def ready(self):
        if False:
            return 10
        f = renpy.test.testfocus.find_focus(self.pattern)
        if f is not None:
            return True
        else:
            return False

class Drag(Node):

    def __init__(self, loc, points):
        if False:
            while True:
                i = 10
        Node.__init__(self, loc)
        self.points = points
        self.pattern = None
        self.button = 1
        self.steps = 10

    def start(self):
        if False:
            for i in range(10):
                print('nop')
        return True

    def execute(self, state, t):
        if False:
            for i in range(10):
                print('nop')
        self.report()
        if renpy.display.interface.trans_pause:
            return state
        if self.pattern:
            f = renpy.test.testfocus.find_focus(self.pattern)
            if f is None:
                return state
        else:
            f = None
        if state is True:
            points = renpy.python.py_eval(self.points)
            points = [renpy.test.testfocus.find_position(f, i) for i in points]
            if len(points) < 2:
                raise Exception('A drag requires at least two points.')
            interpoints = []
            (xa, ya) = points[0]
            interpoints.append((xa, ya))
            for (xb, yb) in points[1:]:
                for i in range(1, self.steps + 1):
                    done = 1.0 * i / self.steps
                    interpoints.append((int(xa + done * (xb - xa)), int(ya + done * (yb - ya))))
                xa = xb
                ya = yb
            (x, y) = interpoints.pop(0)
            renpy.test.testmouse.move_mouse(x, y)
            renpy.test.testmouse.press_mouse(self.button)
        else:
            interpoints = state
            (x, y) = interpoints.pop(0)
            renpy.test.testmouse.move_mouse(x, y)
        if not interpoints:
            renpy.test.testmouse.release_mouse(self.button)
            return None
        else:
            return interpoints

    def ready(self):
        if False:
            return 10
        if self.pattern is None:
            return True
        f = renpy.test.testfocus.find_focus(self.pattern)
        if f is not None:
            return True
        else:
            return False

class Type(Pattern):
    interval = 0.01

    def __init__(self, loc, keys):
        if False:
            i = 10
            return i + 15
        Pattern.__init__(self, loc)
        self.keys = keys

    def start(self):
        if False:
            print('Hello World!')
        return 0

    def perform(self, x, y, state, t):
        if False:
            print('Hello World!')
        if state >= len(self.keys):
            return None
        move_mouse(x, y)
        keysym = self.keys[state]
        renpy.test.testkey.down(self, keysym)
        renpy.test.testkey.up(self, keysym)
        return state + 1

class Action(Node):

    def __init__(self, loc, expr):
        if False:
            print('Hello World!')
        Node.__init__(self, loc)
        self.expr = expr

    def start(self):
        if False:
            for i in range(10):
                print('nop')
        renpy.test.testexecution.action = renpy.python.py_eval(self.expr)
        return True

    def execute(self, state, t):
        if False:
            while True:
                i = 10
        self.report()
        if renpy.test.testexecution.action:
            return True
        else:
            return None

    def ready(self):
        if False:
            print('Hello World!')
        self.report()
        action = renpy.python.py_eval(self.expr)
        return renpy.display.behavior.is_sensitive(action)

class Pause(Node):

    def __init__(self, loc, expr):
        if False:
            for i in range(10):
                print('nop')
        Node.__init__(self, loc)
        self.expr = expr

    def start(self):
        if False:
            i = 10
            return i + 15
        return float(renpy.python.py_eval(self.expr))

    def execute(self, state, t):
        if False:
            i = 10
            return i + 15
        self.report()
        if t < state:
            return state
        else:
            return None

class Label(Node):

    def __init__(self, loc, name):
        if False:
            for i in range(10):
                print('nop')
        Node.__init__(self, loc)
        self.name = name

    def start(self):
        if False:
            return 10
        return True

    def execute(self, state, t):
        if False:
            for i in range(10):
                print('nop')
        if self.name in renpy.test.testexecution.labels:
            return None
        else:
            return state

    def ready(self):
        if False:
            for i in range(10):
                print('nop')
        return self.name in renpy.test.testexecution.labels

class Until(Node):
    """
    Executes `left` repeatedly until `right` is ready, then executes `right`
    once before quitting.
    """

    def __init__(self, loc, left, right):
        if False:
            return 10
        Node.__init__(self, loc)
        self.left = left
        self.right = right

    def start(self):
        if False:
            while True:
                i = 10
        return (None, None, 0)

    def execute(self, state, t):
        if False:
            print('Hello World!')
        (child, child_state, start) = state
        if self.right.ready() and (not child is self.right):
            child = self.right
            child_state = None
        elif child is None:
            child = self.left
        if child_state is None:
            child_state = child.start()
            start = t
        if child_state is not None:
            child_state = child.execute(child_state, t - start)
        if child_state is None and child is self.right:
            return None
        return (child, child_state, start)

    def ready(self):
        if False:
            return 10
        return self.left.ready() or self.right.ready()

class If(Node):
    """
    If `condition` is ready, runs the block. Otherwise, goes to the next
    statement.
    """

    def __init__(self, loc, condition, block):
        if False:
            for i in range(10):
                print('nop')
        Node.__init__(self, loc)
        self.condition = condition
        self.block = block

    def start(self):
        if False:
            while True:
                i = 10
        return (None, None, 0)

    def execute(self, state, t):
        if False:
            for i in range(10):
                print('nop')
        (node, child_state, start) = state
        if node is None:
            if not self.condition.ready():
                return None
            node = self.block
        (node, child_state, start) = renpy.test.testexecution.execute_node(t, node, child_state, start)
        if node is None:
            return None
        return (node, child_state, start)

class Python(Node):

    def __init__(self, loc, code):
        if False:
            for i in range(10):
                print('nop')
        Node.__init__(self, loc)
        self.code = code

    def start(self):
        if False:
            return 10
        renpy.test.testexecution.action = self
        return True

    def execute(self, state, t):
        if False:
            while True:
                i = 10
        self.report()
        if renpy.test.testexecution.action:
            return True
        else:
            return None

    def __call__(self):
        if False:
            while True:
                i = 10
        renpy.python.py_exec_bytecode(self.code.bytecode)

class Assert(Node):

    def __init__(self, loc, expr):
        if False:
            for i in range(10):
                print('nop')
        Node.__init__(self, loc)
        self.expr = expr

    def start(self):
        if False:
            print('Hello World!')
        renpy.test.testexecution.action = self
        return True

    def execute(self, state, t):
        if False:
            while True:
                i = 10
        self.report()
        if renpy.test.testexecution.action:
            return True
        else:
            return None

    def __call__(self):
        if False:
            for i in range(10):
                print('nop')
        if not renpy.python.py_eval(self.expr):
            raise Exception('On line {}:{}, assertion {} failed.'.format(self.filename, self.linenumber, self.expr))

class Jump(Node):

    def __init__(self, loc, target):
        if False:
            i = 10
            return i + 15
        Node.__init__(self, loc)
        self.target = target

    def start(self):
        if False:
            i = 10
            return i + 15
        node = renpy.test.testexecution.lookup(self.target, self)
        raise renpy.test.testexecution.TestJump(node)

class Call(Node):

    def __init__(self, loc, target):
        if False:
            for i in range(10):
                print('nop')
        Node.__init__(self, loc)
        self.target = target

    def start(self):
        if False:
            return 10
        print('Call test', self.target)
        node = renpy.test.testexecution.lookup(self.target, self)
        return (node, None, 0)

    def execute(self, state, t):
        if False:
            while True:
                i = 10
        (node, child_state, start) = state
        (node, child_state, start) = renpy.test.testexecution.execute_node(t, node, child_state, start)
        if node is None:
            return None
        return (node, child_state, start)

class Block(Node):

    def __init__(self, loc, block):
        if False:
            while True:
                i = 10
        Node.__init__(self, loc)
        self.block = block

    def start(self):
        if False:
            for i in range(10):
                print('nop')
        return (0, None, None)

    def execute(self, state, t):
        if False:
            for i in range(10):
                print('nop')
        (i, start, s) = state
        if i >= len(self.block):
            return None
        if s is None:
            s = self.block[i].start()
            start = t
        if s is not None:
            s = self.block[i].execute(s, t - start)
        if s is None:
            i += 1
        return (i, start, s)