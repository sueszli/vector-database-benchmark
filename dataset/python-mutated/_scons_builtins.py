__doc__ = "\nCompatibility idioms for builtins names\n\nThis module adds names to the builtins module for things that we want\nto use in SCons but which don't show up until later Python versions than\nthe earliest ones we support.\n\nThis module checks for the following builtins names:\n\n        all()\n        any()\n        memoryview()\n\nImplementations of functions are *NOT* guaranteed to be fully compliant\nwith these functions in later versions of Python.  We are only concerned\nwith adding functionality that we actually use in SCons, so be wary\nif you lift this code for other uses.  (That said, making these more\nnearly the same as later, official versions is still a desirable goal,\nwe just don't need to be obsessive about it.)\n\nIf you're looking at this with pydoc and various names don't show up in\nthe FUNCTIONS or DATA output, that means those names are already built in\nto this version of Python and we don't need to add them from this module.\n"
__revision__ = 'src/engine/SCons/compat/_scons_builtins.py  2014/07/05 09:42:21 garyo'
import builtins
try:
    all
except NameError:

    def all(iterable):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns True if all elements of the iterable are true.\n        '
        for element in iterable:
            if not element:
                return False
        return True
    builtins.all = all
    all = all
try:
    any
except NameError:

    def any(iterable):
        if False:
            i = 10
            return i + 15
        '\n        Returns True if any element of the iterable is true.\n        '
        for element in iterable:
            if element:
                return True
        return False
    builtins.any = any
    any = any
try:
    memoryview
except NameError:

    class memoryview(object):

        def __init__(self, obj):
            if False:
                i = 10
                return i + 15
            self.obj = buffer(obj)

        def __getitem__(self, indx):
            if False:
                i = 10
                return i + 15
            if isinstance(indx, slice):
                return self.obj[indx.start:indx.stop]
            else:
                return self.obj[indx]
    builtins.memoryview = memoryview