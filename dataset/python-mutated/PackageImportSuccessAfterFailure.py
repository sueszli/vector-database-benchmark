from __future__ import print_function
ORIG = None

def display_difference(dct):
    if False:
        print('Hello World!')
    print(ORIG.symmetric_difference(dct))
if str is bytes:
    e = None
ORIG = set(globals())
print('Initial try on top level package:')
try:
    import variable_package
except BaseException as e:
    print('Occurred', str(e))
display_difference(globals())
print('Retry with submodule import:')
try:
    from variable_package.SomeModule import Class4
except BaseException as e:
    print('Occurred', str(e))
display_difference(globals())
print('Try with import from submodule:')
try:
    from variable_package import SomeModule
except BaseException as e:
    print('Occurred', str(e))
display_difference(globals())
print('Try with variable import from top level package assigned before raise:')
try:
    from variable_package import raisy
except BaseException as e:
    print('Occurred', str(e))
display_difference(globals())
print('Try with variable import from top level package assigned after raise:')
try:
    from variable_package import Class5
except BaseException as e:
    print('Occurred', str(e))
display_difference(globals())
print('Try with variable import from top level package assigned before raise:')
try:
    from variable_package import Class3
except BaseException as e:
    print('Occurred', str(e))
display_difference(globals().keys())