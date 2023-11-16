"""Python level PGO handling in Nuitka."""
import os
import struct
from nuitka.__past__ import xrange
from nuitka.Options import getPythonPgoUnseenModulePolicy
from nuitka.Tracing import pgo_logger
_pgo_active = False
_pgo_strings = None
_module_entries = {}
_module_exits = {}

def _readCString(input_file):
    if False:
        while True:
            i = 10
    return b''.join(iter(lambda : input_file.read(1), b'\x00'))

def _readCIntValue(input_file):
    if False:
        for i in range(10):
            print('nop')
    return struct.unpack('i', input_file.read(4))[0]

def _readStringValue(input_file):
    if False:
        while True:
            i = 10
    return _pgo_strings[_readCIntValue(input_file)]

def readPGOInputFile(input_filename):
    if False:
        print('Hello World!')
    'Read PGO information produced by a PGO run.'
    global _pgo_strings, _pgo_active
    with open(input_filename, 'rb') as input_file:
        header = input_file.read(7)
        if header != b'KAY.PGO':
            pgo_logger.sysexit("Error, file '%s' is not a valid PGO input for this version of Nuitka." % input_filename)
        input_file.seek(-7, os.SEEK_END)
        header = input_file.read(7)
        if header != b'YAK.PGO':
            pgo_logger.sysexit("Error, file '%s' was not completed correctly." % input_filename)
        input_file.seek(-8 - 7, os.SEEK_END)
        (count, offset) = struct.unpack('ii', input_file.read(8))
        input_file.seek(offset, os.SEEK_SET)
        _pgo_strings = [None] * count
        for i in xrange(count):
            _pgo_strings[i] = _readCString(input_file)
        input_file.seek(7, os.SEEK_SET)
        while True:
            probe_name = _readStringValue(input_file)
            if probe_name == 'ModuleEnter':
                module_name = _readStringValue(input_file)
                arg = _readCIntValue(input_file)
                _module_entries[module_name] = arg
            elif probe_name == 'ModuleExit':
                module_name = _readStringValue(input_file)
                had_error = _readCIntValue(input_file) != 0
                _module_exits[module_name] = had_error
            elif probe_name == 'END':
                break
            else:
                pgo_logger.sysexit("Error, unknown problem '%s' encountered." % probe_name)
    _pgo_active = True

def decideInclusionFromPGO(module_name, module_kind):
    if False:
        print('Hello World!')
    "Decide module inclusion based on PGO input.\n\n    At this point, PGO can decide the inclusion to not be done. It will\n    ask to include things it has seen at run time, and that won't be a\n    problem, but it will ask to exclude modules not seen entered at runtime,\n    the decision for bytecode is same as inclusion, but the demotion is done\n    later, after first compiling it. Caching might save compile time a second\n    time around once the cache is populated, but care must be taken for that\n    to not cause inclusions that are not used.\n    "
    if not _pgo_active:
        return None
    if module_kind == 'extension':
        return None
    if module_name in _module_entries:
        return True
    unseen_module_policy = getPythonPgoUnseenModulePolicy()
    if unseen_module_policy == 'exclude':
        return False
    else:
        return None

def decideCompilationFromPGO(module_name):
    if False:
        while True:
            i = 10
    if not _pgo_active:
        return None
    unseen_module_policy = getPythonPgoUnseenModulePolicy()
    if module_name not in _module_entries and unseen_module_policy == 'bytecode':
        return 'bytecode'
    else:
        return None