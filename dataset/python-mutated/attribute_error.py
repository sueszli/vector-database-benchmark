class Cls:
    class_attr = ''

    def __init__(self, input):
        if False:
            i = 10
            return i + 15
        self.instance_attr = 3
        self.input = input

    def f(self):
        if False:
            print('Hello World!')
        return self.not_existing

    def undefined_object(self, obj):
        if False:
            for i in range(10):
                print('nop')
        "\n        Uses an arbitrary object and performs an operation on it, shouldn't\n        be a problem.\n        "
        obj.arbitrary_lookup

    def defined_lookup(self, obj):
        if False:
            while True:
                i = 10
        '\n        `obj` is defined by a call into this function.\n        '
        obj.upper
        obj.arbitrary_lookup
    class_attr = a
Cls(1).defined_lookup('')
c = Cls(1)
c.class_attr
Cls.class_attr
Cls.class_attr_error
c.instance_attr
c.instance_attr_error
c.something = None
something = a
something
for loop_variable in [1, 2]:
    x = undefined
    loop_variable
for loop_variable in [1, 2, undefined]:
    pass
[1, ''.undefined_attr]

def return_one(something):
    if False:
        return 10
    return 1
return_one(''.undefined_attribute)
[r for r in undefined]
[undefined for r in [1, 2]]
[r for r in [1, 2]]

class NotCalled:

    def match_something(self, param):
        if False:
            while True:
                i = 10
        seems_to_need_an_assignment = param
        return [value.match_something() for value in []]

@undefined_decorator
def func():
    if False:
        for i in range(10):
            print('nop')
    return 1
string = '%s %s' % (1, 2)
string.upper
import import_tree
import_tree.a
import_tree.b