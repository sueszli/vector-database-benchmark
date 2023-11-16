__revision__ = 'src/engine/SCons/Sig.py  2014/07/05 09:42:21 garyo'
__doc__ = "Place-holder for the old SCons.Sig module hierarchy\n\nThis is no longer used, but code out there (such as the NSIS module on\nthe SCons wiki) may try to import SCons.Sig.  If so, we generate a warning\nthat points them to the line that caused the import, and don't die.\n\nIf someone actually tried to use the sub-modules or functions within\nthe package (for example, SCons.Sig.MD5.signature()), then they'll still\nget an AttributeError, but at least they'll know where to start looking.\n"
import SCons.Util
import SCons.Warnings
msg = 'The SCons.Sig module no longer exists.\n    Remove the following "import SCons.Sig" line to eliminate this warning:'
SCons.Warnings.warn(SCons.Warnings.DeprecatedSigModuleWarning, msg)
default_calc = None
default_module = None

class MD5Null(SCons.Util.Null):

    def __repr__(self):
        if False:
            while True:
                i = 10
        return 'MD5Null()'

class TimeStampNull(SCons.Util.Null):

    def __repr__(self):
        if False:
            while True:
                i = 10
        return 'TimeStampNull()'
MD5 = MD5Null()
TimeStamp = TimeStampNull()