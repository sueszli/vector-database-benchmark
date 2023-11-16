__revision__ = 'src/engine/SCons/Options/ListOption.py  2014/07/05 09:42:21 garyo'
__doc__ = 'Place-holder for the old SCons.Options module hierarchy\n\nThis is for backwards compatibility.  The new equivalent is the Variables/\nclass hierarchy.  These will have deprecation warnings added (some day),\nand will then be removed entirely (some day).\n'
import SCons.Variables
import SCons.Warnings
warned = False

def ListOption(*args, **kw):
    if False:
        for i in range(10):
            print('nop')
    global warned
    if not warned:
        msg = 'The ListOption() function is deprecated; use the ListVariable() function instead.'
        SCons.Warnings.warn(SCons.Warnings.DeprecatedOptionsWarning, msg)
        warned = True
    return SCons.Variables.ListVariable(*args, **kw)