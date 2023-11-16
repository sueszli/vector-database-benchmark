__revision__ = 'src/engine/SCons/Options/PackageOption.py  2014/07/05 09:42:21 garyo'
__doc__ = 'Place-holder for the old SCons.Options module hierarchy\n\nThis is for backwards compatibility.  The new equivalent is the Variables/\nclass hierarchy.  These will have deprecation warnings added (some day),\nand will then be removed entirely (some day).\n'
import SCons.Variables
import SCons.Warnings
warned = False

def PackageOption(*args, **kw):
    if False:
        print('Hello World!')
    global warned
    if not warned:
        msg = 'The PackageOption() function is deprecated; use the PackageVariable() function instead.'
        SCons.Warnings.warn(SCons.Warnings.DeprecatedOptionsWarning, msg)
        warned = True
    return SCons.Variables.PackageVariable(*args, **kw)