__revision__ = 'src/engine/SCons/Options/EnumOption.py  2014/07/05 09:42:21 garyo'
__doc__ = 'Place-holder for the old SCons.Options module hierarchy\n\nThis is for backwards compatibility.  The new equivalent is the Variables/\nclass hierarchy.  These will have deprecation warnings added (some day),\nand will then be removed entirely (some day).\n'
import SCons.Variables
import SCons.Warnings
warned = False

def EnumOption(*args, **kw):
    if False:
        i = 10
        return i + 15
    global warned
    if not warned:
        msg = 'The EnumOption() function is deprecated; use the EnumVariable() function instead.'
        SCons.Warnings.warn(SCons.Warnings.DeprecatedOptionsWarning, msg)
        warned = True
    return SCons.Variables.EnumVariable(*args, **kw)