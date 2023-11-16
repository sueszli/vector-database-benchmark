from __future__ import print_function
import timeit

class TreeNode(object):

    def __init__(self, name):
        if False:
            for i in range(10):
                print('nop')
        self._name = name
        self._children = []
        self._time = 0

    def __str__(self):
        if False:
            return 10
        return '%s: %s' % (self._name, self._time)
    __repr__ = __str__

    def add_child(self, node):
        if False:
            while True:
                i = 10
        self._children.append(node)

    def children(self):
        if False:
            i = 10
            return i + 15
        return self._children

    def child(self, i):
        if False:
            while True:
                i = 10
        return self.children()[i]

    def set_time(self, time):
        if False:
            i = 10
            return i + 15
        self._time = time

    def time(self):
        if False:
            print('Hello World!')
        return self._time
    total_time = time

    def exclusive_time(self):
        if False:
            print('Hello World!')
        return self.total_time() - sum((child.time() for child in self.children()))

    def name(self):
        if False:
            i = 10
            return i + 15
        return self._name

    def linearize(self):
        if False:
            print('Hello World!')
        res = [self]
        for child in self.children():
            res.extend(child.linearize())
        return res

    def print_tree(self, level=0, max_depth=None):
        if False:
            while True:
                i = 10
        print('  ' * level + str(self))
        if max_depth is not None and max_depth <= level:
            return
        for child in self.children():
            child.print_tree(level + 1, max_depth=max_depth)

    def print_generic(self, n=50, method='time'):
        if False:
            i = 10
            return i + 15
        slowest = sorted(((getattr(node, method)(), node.name()) for node in self.linearize()))[-n:]
        for (time, name) in slowest[::-1]:
            print('%s %s' % (time, name))

    def print_slowest(self, n=50):
        if False:
            while True:
                i = 10
        self.print_generic(n=50, method='time')

    def print_slowest_exclusive(self, n=50):
        if False:
            return 10
        self.print_generic(n, method='exclusive_time')

    def write_cachegrind(self, f):
        if False:
            return 10
        if isinstance(f, str):
            f = open(f, 'w')
            f.write('events: Microseconds\n')
            f.write('fl=sympyallimport\n')
            must_close = True
        else:
            must_close = False
        f.write('fn=%s\n' % self.name())
        f.write('1 %s\n' % self.exclusive_time())
        counter = 2
        for child in self.children():
            f.write('cfn=%s\n' % child.name())
            f.write('calls=1 1\n')
            f.write('%s %s\n' % (counter, child.time()))
            counter += 1
        f.write('\n\n')
        for child in self.children():
            child.write_cachegrind(f)
        if must_close:
            f.close()
pp = TreeNode(None)
seen = set()

def new_import(name, globals={}, locals={}, fromlist=[]):
    if False:
        i = 10
        return i + 15
    global pp
    if name in seen:
        return old_import(name, globals, locals, fromlist)
    seen.add(name)
    node = TreeNode(name)
    pp.add_child(node)
    old_pp = pp
    pp = node
    t1 = timeit.default_timer()
    module = old_import(name, globals, locals, fromlist)
    t2 = timeit.default_timer()
    node.set_time(int(1000000 * (t2 - t1)))
    pp = old_pp
    return module
old_import = __builtins__.__import__
__builtins__.__import__ = new_import
old_sum = sum
from sympy import *
sum = old_sum
sageall = pp.child(0)
sageall.write_cachegrind('sympy.cachegrind')
print('Timings saved. Do:\n$ kcachegrind sympy.cachegrind')