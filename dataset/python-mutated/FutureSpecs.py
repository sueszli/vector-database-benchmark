""" Specification record for future flags.

A source reference also implies a specific set of future flags in use by the
parser at that location. Can be different inside a module due to e.g. the
in-lining of "exec" statements with their own future imports, or in-lining of
code from other modules.
"""
from nuitka.PythonVersions import python_version
from nuitka.utils.InstanceCounters import counted_del, counted_init, isCountingInstances
_future_division_default = python_version >= 768
_future_absolute_import_default = python_version >= 768
_future_generator_stop_default = python_version >= 880
_future_annotations_default = python_version >= 1024

class FutureSpec(object):
    __slots__ = ('future_division', 'unicode_literals', 'absolute_import', 'future_print', 'barry_bdfl', 'generator_stop', 'future_annotations')

    @counted_init
    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.future_division = _future_division_default
        self.unicode_literals = False
        self.absolute_import = _future_absolute_import_default
        self.future_print = False
        self.barry_bdfl = False
        self.generator_stop = _future_generator_stop_default
        self.future_annotations = _future_annotations_default
    if isCountingInstances():
        __del__ = counted_del()

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '<FutureSpec %s>' % ','.join(self.asFlags())

    def clone(self):
        if False:
            for i in range(10):
                print('nop')
        result = FutureSpec()
        result.future_division = self.future_division
        result.unicode_literals = self.unicode_literals
        result.absolute_import = self.absolute_import
        result.future_print = self.future_print
        result.barry_bdfl = self.barry_bdfl
        result.generator_stop = self.generator_stop
        result.future_annotations = result.future_annotations
        return result

    def isFutureDivision(self):
        if False:
            return 10
        return self.future_division

    def enableFutureDivision(self):
        if False:
            print('Hello World!')
        self.future_division = True

    def isFuturePrint(self):
        if False:
            i = 10
            return i + 15
        return self.future_print

    def enableFuturePrint(self):
        if False:
            i = 10
            return i + 15
        self.future_print = True

    def enableUnicodeLiterals(self):
        if False:
            return 10
        self.unicode_literals = True

    def enableAbsoluteImport(self):
        if False:
            print('Hello World!')
        self.absolute_import = True

    def enableBarry(self):
        if False:
            while True:
                i = 10
        self.barry_bdfl = True

    def enableGeneratorStop(self):
        if False:
            print('Hello World!')
        self.generator_stop = True

    def isAbsoluteImport(self):
        if False:
            print('Hello World!')
        return self.absolute_import

    def isGeneratorStop(self):
        if False:
            i = 10
            return i + 15
        return self.generator_stop

    def enableFutureAnnotations(self):
        if False:
            print('Hello World!')
        self.future_annotations = True

    def isFutureAnnotations(self):
        if False:
            i = 10
            return i + 15
        return self.future_annotations

    def asFlags(self):
        if False:
            while True:
                i = 10
        'Create a list of C identifiers to represent the flag values.\n\n        This is for use in code generation and to restore from\n        saved modules.\n        '
        result = []
        if python_version < 768 and self.future_division:
            result.append('CO_FUTURE_DIVISION')
        if self.unicode_literals:
            result.append('CO_FUTURE_UNICODE_LITERALS')
        if python_version < 768 and self.absolute_import:
            result.append('CO_FUTURE_ABSOLUTE_IMPORT')
        if python_version < 768 and self.future_print:
            result.append('CO_FUTURE_PRINT_FUNCTION')
        if python_version >= 768 and self.barry_bdfl:
            result.append('CO_FUTURE_BARRY_AS_BDFL')
        if 848 <= python_version < 880 and self.generator_stop:
            result.append('CO_FUTURE_GENERATOR_STOP')
        if python_version >= 880 and self.future_annotations:
            result.append('CO_FUTURE_ANNOTATIONS')
        return tuple(result)

def fromFlags(flags):
    if False:
        return 10
    flags = flags.split(',')
    if '' in flags:
        flags.remove('')
    result = FutureSpec()
    if 'CO_FUTURE_DIVISION' in flags:
        result.enableFutureDivision()
    if 'CO_FUTURE_UNICODE_LITERALS' in flags:
        result.enableUnicodeLiterals()
    if 'CO_FUTURE_ABSOLUTE_IMPORT' in flags:
        result.enableAbsoluteImport()
    if 'CO_FUTURE_PRINT_FUNCTION' in flags:
        result.enableFuturePrint()
    if 'CO_FUTURE_BARRY_AS_BDFL' in flags:
        result.enableBarry()
    if 'CO_FUTURE_GENERATOR_STOP' in flags:
        result.enableGeneratorStop()
    assert tuple(result.asFlags()) == tuple(flags), (result, result.asFlags(), flags)
    return result