"""SCons.Warnings

This file implements the warnings framework for SCons.

"""
__revision__ = 'src/engine/SCons/Warnings.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
import sys
import SCons.Errors

class Warning(SCons.Errors.UserError):
    pass

class WarningOnByDefault(Warning):
    pass

class TargetNotBuiltWarning(Warning):
    pass

class CacheVersionWarning(WarningOnByDefault):
    pass

class CacheWriteErrorWarning(Warning):
    pass

class CorruptSConsignWarning(WarningOnByDefault):
    pass

class DependencyWarning(Warning):
    pass

class DevelopmentVersionWarning(WarningOnByDefault):
    pass

class DuplicateEnvironmentWarning(WarningOnByDefault):
    pass

class FutureReservedVariableWarning(WarningOnByDefault):
    pass

class LinkWarning(WarningOnByDefault):
    pass

class MisleadingKeywordsWarning(WarningOnByDefault):
    pass

class MissingSConscriptWarning(WarningOnByDefault):
    pass

class NoObjectCountWarning(WarningOnByDefault):
    pass

class NoParallelSupportWarning(WarningOnByDefault):
    pass

class ReservedVariableWarning(WarningOnByDefault):
    pass

class StackSizeWarning(WarningOnByDefault):
    pass

class VisualCMissingWarning(WarningOnByDefault):
    pass

class VisualVersionMismatch(WarningOnByDefault):
    pass

class VisualStudioMissingWarning(Warning):
    pass

class FortranCxxMixWarning(LinkWarning):
    pass

class FutureDeprecatedWarning(Warning):
    pass

class DeprecatedWarning(Warning):
    pass

class MandatoryDeprecatedWarning(DeprecatedWarning):
    pass

class PythonVersionWarning(DeprecatedWarning):
    pass

class DeprecatedSourceCodeWarning(FutureDeprecatedWarning):
    pass

class TaskmasterNeedsExecuteWarning(DeprecatedWarning):
    pass

class DeprecatedOptionsWarning(MandatoryDeprecatedWarning):
    pass

class DeprecatedDebugOptionsWarning(MandatoryDeprecatedWarning):
    pass

class DeprecatedMissingSConscriptWarning(DeprecatedWarning):
    pass
_enabled = []
_warningAsException = 0
_warningOut = None

def suppressWarningClass(clazz):
    if False:
        print('Hello World!')
    'Suppresses all warnings that are of type clazz or\n    derived from clazz.'
    _enabled.insert(0, (clazz, 0))

def enableWarningClass(clazz):
    if False:
        for i in range(10):
            print('nop')
    'Enables all warnings that are of type clazz or\n    derived from clazz.'
    _enabled.insert(0, (clazz, 1))

def warningAsException(flag=1):
    if False:
        return 10
    'Turn warnings into exceptions.  Returns the old value of the flag.'
    global _warningAsException
    old = _warningAsException
    _warningAsException = flag
    return old

def warn(clazz, *args):
    if False:
        while True:
            i = 10
    global _enabled, _warningAsException, _warningOut
    warning = clazz(args)
    for (cls, flag) in _enabled:
        if isinstance(warning, cls):
            if flag:
                if _warningAsException:
                    raise warning
                if _warningOut:
                    _warningOut(warning)
            break

def process_warn_strings(arguments):
    if False:
        while True:
            i = 10
    'Process requests to enable/disable warnings.\n\n    The requests are strings passed to the --warn option or the\n    SetOption(\'warn\') function.\n\n    An argument to this option should be of the form <warning-class>\n    or no-<warning-class>.  The warning class is munged in order\n    to get an actual class name from the classes above, which we\n    need to pass to the {enable,disable}WarningClass() functions.\n    The supplied <warning-class> is split on hyphens, each element\n    is capitalized, then smushed back together.  Then the string\n    "Warning" is appended to get the class name.\n\n    For example, \'deprecated\' will enable the DeprecatedWarning\n    class.  \'no-dependency\' will disable the DependencyWarning class.\n\n    As a special case, --warn=all and --warn=no-all will enable or\n    disable (respectively) the base Warning class of all warnings.\n    '

    def _capitalize(s):
        if False:
            i = 10
            return i + 15
        if s[:5] == 'scons':
            return 'SCons' + s[5:]
        else:
            return s.capitalize()
    for arg in arguments:
        elems = arg.lower().split('-')
        enable = 1
        if elems[0] == 'no':
            enable = 0
            del elems[0]
        if len(elems) == 1 and elems[0] == 'all':
            class_name = 'Warning'
        else:
            class_name = ''.join(map(_capitalize, elems)) + 'Warning'
        try:
            clazz = globals()[class_name]
        except KeyError:
            sys.stderr.write("No warning type: '%s'\n" % arg)
        else:
            if enable:
                enableWarningClass(clazz)
            elif issubclass(clazz, MandatoryDeprecatedWarning):
                fmt = "Can not disable mandataory warning: '%s'\n"
                sys.stderr.write(fmt % arg)
            else:
                suppressWarningClass(clazz)