__revision__ = 'src/engine/SCons/Options/PathOption.py  2014/07/05 09:42:21 garyo'
__doc__ = 'Place-holder for the old SCons.Options module hierarchy\n\nThis is for backwards compatibility.  The new equivalent is the Variables/\nclass hierarchy.  These will have deprecation warnings added (some day),\nand will then be removed entirely (some day).\n'
import SCons.Variables
import SCons.Warnings
warned = False

class _PathOptionClass(object):

    def warn(self):
        if False:
            for i in range(10):
                print('nop')
        global warned
        if not warned:
            msg = 'The PathOption() function is deprecated; use the PathVariable() function instead.'
            SCons.Warnings.warn(SCons.Warnings.DeprecatedOptionsWarning, msg)
            warned = True

    def __call__(self, *args, **kw):
        if False:
            for i in range(10):
                print('nop')
        self.warn()
        return SCons.Variables.PathVariable(*args, **kw)

    def PathAccept(self, *args, **kw):
        if False:
            for i in range(10):
                print('nop')
        self.warn()
        return SCons.Variables.PathVariable.PathAccept(*args, **kw)

    def PathIsDir(self, *args, **kw):
        if False:
            return 10
        self.warn()
        return SCons.Variables.PathVariable.PathIsDir(*args, **kw)

    def PathIsDirCreate(self, *args, **kw):
        if False:
            for i in range(10):
                print('nop')
        self.warn()
        return SCons.Variables.PathVariable.PathIsDirCreate(*args, **kw)

    def PathIsFile(self, *args, **kw):
        if False:
            print('Hello World!')
        self.warn()
        return SCons.Variables.PathVariable.PathIsFile(*args, **kw)

    def PathExists(self, *args, **kw):
        if False:
            for i in range(10):
                print('nop')
        self.warn()
        return SCons.Variables.PathVariable.PathExists(*args, **kw)
PathOption = _PathOptionClass()