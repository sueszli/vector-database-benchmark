""" Code generation contexts.

"""
import collections
from abc import abstractmethod
from contextlib import contextmanager
from nuitka import Options
from nuitka.__past__ import iterItems
from nuitka.Constants import isMutable
from nuitka.PythonVersions import python_version
from nuitka.Serialization import ConstantAccessor
from nuitka.utils.Hashing import getStringHash
from nuitka.utils.InstanceCounters import counted_del, counted_init, isCountingInstances
from nuitka.utils.SlotMetaClasses import getMetaClassBase
from .VariableDeclarations import VariableDeclaration, VariableStorage

class TempMixin(object):
    __slots__ = ()

    def __init__(self):
        if False:
            return 10
        self.tmp_names = {}
        self.labels = {}
        self.exception_escape = None
        self.loop_continue = None
        self.loop_break = None
        self.true_target = None
        self.false_target = None
        self.keeper_variable_count = 0
        self.exception_keepers = (None, None, None, None)
        self.preserver_variable_declaration = {}
        self.cleanup_names = []

    def _formatTempName(self, base_name, number):
        if False:
            print('Hello World!')
        if number is None:
            return 'tmp_{name}'.format(name=base_name)
        else:
            return 'tmp_{name}_{number:d}'.format(name=base_name, number=number)

    def allocateTempName(self, base_name, type_name='PyObject *', unique=False):
        if False:
            for i in range(10):
                print('nop')
        if unique:
            number = None
        else:
            number = self.tmp_names.get(base_name, 0)
            number += 1
        self.tmp_names[base_name] = number
        formatted_name = self._formatTempName(base_name=base_name, number=number)
        if unique:
            result = self.variable_storage.getVariableDeclarationTop(formatted_name)
            if result is None:
                if base_name == 'outline_return_value':
                    init_value = 'NULL'
                elif base_name == 'return_value':
                    init_value = 'NULL'
                elif base_name == 'generator_return':
                    init_value = 'false'
                else:
                    init_value = None
                if base_name == 'unused':
                    result = self.variable_storage.addVariableDeclarationFunction(c_type=type_name, code_name=formatted_name, init_value=init_value)
                else:
                    result = self.variable_storage.addVariableDeclarationTop(c_type=type_name, code_name=formatted_name, init_value=init_value)
            else:
                if type_name.startswith('NUITKA_MAY_BE_UNUSED'):
                    type_name = type_name[21:]
                assert result.c_type == type_name
        else:
            result = self.variable_storage.addVariableDeclarationLocal(c_type=type_name, code_name=formatted_name)
        return result

    def skipTempName(self, base_name):
        if False:
            return 10
        number = self.tmp_names.get(base_name, 0)
        number += 1
        self.tmp_names[base_name] = number

    def getIntResName(self):
        if False:
            while True:
                i = 10
        return self.allocateTempName('res', 'int', unique=True)

    def getBoolResName(self):
        if False:
            for i in range(10):
                print('nop')
        return self.allocateTempName('result', 'bool', unique=True)

    def hasTempName(self, base_name):
        if False:
            for i in range(10):
                print('nop')
        return base_name in self.tmp_names

    def getExceptionEscape(self):
        if False:
            print('Hello World!')
        return self.exception_escape

    def setExceptionEscape(self, label):
        if False:
            return 10
        result = self.exception_escape
        self.exception_escape = label
        return result

    def getLoopBreakTarget(self):
        if False:
            return 10
        return self.loop_break

    def setLoopBreakTarget(self, label):
        if False:
            while True:
                i = 10
        result = self.loop_break
        self.loop_break = label
        return result

    def getLoopContinueTarget(self):
        if False:
            print('Hello World!')
        return self.loop_continue

    def setLoopContinueTarget(self, label):
        if False:
            print('Hello World!')
        result = self.loop_continue
        self.loop_continue = label
        return result

    def allocateLabel(self, label):
        if False:
            return 10
        result = self.labels.get(label, 0)
        result += 1
        self.labels[label] = result
        return '{name}_{number:d}'.format(name=label, number=result)

    def getLabelCount(self, label):
        if False:
            i = 10
            return i + 15
        return self.labels.get(label, 0)

    def allocateExceptionKeeperVariables(self):
        if False:
            i = 10
            return i + 15
        self.keeper_variable_count += 1
        debug = Options.is_debug and python_version >= 768
        if debug:
            keeper_obj_init = 'NULL'
        else:
            keeper_obj_init = None
        return (self.variable_storage.addVariableDeclarationTop('PyObject *', 'exception_keeper_type_%d' % self.keeper_variable_count, keeper_obj_init), self.variable_storage.addVariableDeclarationTop('PyObject *', 'exception_keeper_value_%d' % self.keeper_variable_count, keeper_obj_init), self.variable_storage.addVariableDeclarationTop('PyTracebackObject *', 'exception_keeper_tb_%d' % self.keeper_variable_count, keeper_obj_init), self.variable_storage.addVariableDeclarationTop('NUITKA_MAY_BE_UNUSED int', 'exception_keeper_lineno_%d' % self.keeper_variable_count, '0' if debug else None))

    def getExceptionKeeperVariables(self):
        if False:
            i = 10
            return i + 15
        return self.exception_keepers

    def setExceptionKeeperVariables(self, keeper_vars):
        if False:
            for i in range(10):
                print('nop')
        result = self.exception_keepers
        self.exception_keepers = tuple(keeper_vars)
        return result

    def addExceptionPreserverVariables(self, preserver_id):
        if False:
            return 10
        if preserver_id not in self.preserver_variable_declaration:
            needs_init = Options.is_debug and python_version >= 768
            if needs_init:
                preserver_obj_init = 'Nuitka_ExceptionStackItem_Empty'
            else:
                preserver_obj_init = None
            self.preserver_variable_declaration[preserver_id] = self.variable_storage.addVariableDeclarationTop('struct Nuitka_ExceptionStackItem', 'exception_preserved_%d' % preserver_id, preserver_obj_init)
        return self.preserver_variable_declaration[preserver_id]

    def getTrueBranchTarget(self):
        if False:
            return 10
        return self.true_target

    def getFalseBranchTarget(self):
        if False:
            while True:
                i = 10
        return self.false_target

    def setTrueBranchTarget(self, label):
        if False:
            for i in range(10):
                print('nop')
        self.true_target = label

    def setFalseBranchTarget(self, label):
        if False:
            i = 10
            return i + 15
        self.false_target = label

    def getCleanupTempNames(self):
        if False:
            print('Hello World!')
        return self.cleanup_names[-1]

    def addCleanupTempName(self, tmp_name):
        if False:
            print('Hello World!')
        assert tmp_name not in self.cleanup_names[-1], tmp_name
        assert tmp_name.c_type != 'nuitka_void' or tmp_name.code_name == 'tmp_unused', tmp_name
        self.cleanup_names[-1].append(tmp_name)

    def removeCleanupTempName(self, tmp_name):
        if False:
            print('Hello World!')
        assert tmp_name in self.cleanup_names[-1], tmp_name
        self.cleanup_names[-1].remove(tmp_name)

    def transferCleanupTempName(self, tmp_source, tmp_dest):
        if False:
            print('Hello World!')
        if self.needsCleanup(tmp_source):
            self.addCleanupTempName(tmp_dest)
            self.removeCleanupTempName(tmp_source)

    def needsCleanup(self, tmp_name):
        if False:
            print('Hello World!')
        return tmp_name in self.cleanup_names[-1]

    def pushCleanupScope(self):
        if False:
            print('Hello World!')
        self.cleanup_names.append([])

    def popCleanupScope(self):
        if False:
            while True:
                i = 10
        assert not self.cleanup_names[-1]
        del self.cleanup_names[-1]
CodeObjectHandle = collections.namedtuple('CodeObjectHandle', ('co_name', 'co_qualname', 'co_kind', 'co_varnames', 'co_argcount', 'co_posonlyargcount', 'co_kwonlyargcount', 'co_has_starlist', 'co_has_stardict', 'co_filename', 'line_number', 'future_flags', 'co_new_locals', 'co_freevars', 'is_optimized'))

class CodeObjectsMixin(object):
    __slots__ = ()

    def __init__(self):
        if False:
            while True:
                i = 10
        self.code_objects = {}

    def getCodeObjects(self):
        if False:
            for i in range(10):
                print('nop')
        return sorted(iterItems(self.code_objects))

    def getCodeObjectHandle(self, code_object):
        if False:
            for i in range(10):
                print('nop')
        key = CodeObjectHandle(co_filename=code_object.getFilename(), co_name=code_object.getCodeObjectName(), co_qualname=code_object.getCodeObjectQualname(), line_number=code_object.getLineNumber(), co_varnames=code_object.getVarNames(), co_argcount=code_object.getArgumentCount(), co_freevars=code_object.getFreeVarNames(), co_posonlyargcount=code_object.getPosOnlyParameterCount(), co_kwonlyargcount=code_object.getKwOnlyParameterCount(), co_kind=code_object.getCodeObjectKind(), is_optimized=code_object.getFlagIsOptimizedValue(), co_new_locals=code_object.getFlagNewLocalsValue(), co_has_starlist=code_object.hasStarListArg(), co_has_stardict=code_object.hasStarDictArg(), future_flags=code_object.getFutureSpec().asFlags())
        if key not in self.code_objects:
            self.code_objects[key] = 'codeobj_%s' % self._calcHash(key)
        return self.code_objects[key]

    def _calcHash(self, key):
        if False:
            while True:
                i = 10
        return getStringHash('-'.join((str(s) for s in key)))

class PythonContextBase(getMetaClassBase('Context', require_slots=True)):
    __slots__ = ('source_ref', 'current_source_ref')

    @counted_init
    def __init__(self):
        if False:
            return 10
        self.source_ref = None
        self.current_source_ref = None
    if isCountingInstances():
        __del__ = counted_del()

    def getCurrentSourceCodeReference(self):
        if False:
            while True:
                i = 10
        return self.current_source_ref

    def setCurrentSourceCodeReference(self, value):
        if False:
            print('Hello World!')
        result = self.current_source_ref
        self.current_source_ref = value
        return result

    @contextmanager
    def withCurrentSourceCodeReference(self, value):
        if False:
            while True:
                i = 10
        old_source_ref = self.setCurrentSourceCodeReference(value)
        yield old_source_ref
        self.setCurrentSourceCodeReference(value)

    def getInplaceLeftName(self):
        if False:
            while True:
                i = 10
        return self.allocateTempName('inplace_orig', 'PyObject *', True)

    @abstractmethod
    def getConstantCode(self, constant, deep_check=False):
        if False:
            for i in range(10):
                print('nop')
        pass

    @abstractmethod
    def getModuleCodeName(self):
        if False:
            i = 10
            return i + 15
        pass

    @abstractmethod
    def getModuleName(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @abstractmethod
    def addHelperCode(self, key, code):
        if False:
            return 10
        pass

    @abstractmethod
    def hasHelperCode(self, key):
        if False:
            print('Hello World!')
        pass

    @abstractmethod
    def addDeclaration(self, key, code):
        if False:
            for i in range(10):
                print('nop')
        pass

    @abstractmethod
    def pushFrameVariables(self, frame_variables):
        if False:
            while True:
                i = 10
        pass

    @abstractmethod
    def popFrameVariables(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @abstractmethod
    def getFrameVariableTypeDescriptions(self):
        if False:
            print('Hello World!')
        pass

    @abstractmethod
    def getFrameVariableTypeDescription(self):
        if False:
            while True:
                i = 10
        pass

    @abstractmethod
    def getFrameTypeDescriptionDeclaration(self):
        if False:
            return 10
        pass

    @abstractmethod
    def getFrameVariableCodeNames(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @abstractmethod
    def allocateTempName(self, base_name, type_name='PyObject *', unique=False):
        if False:
            print('Hello World!')
        pass

    @abstractmethod
    def skipTempName(self, base_name):
        if False:
            while True:
                i = 10
        pass

    @abstractmethod
    def getIntResName(self):
        if False:
            while True:
                i = 10
        pass

    @abstractmethod
    def getBoolResName(self):
        if False:
            return 10
        pass

    @abstractmethod
    def hasTempName(self, base_name):
        if False:
            print('Hello World!')
        pass

    @abstractmethod
    def getExceptionEscape(self):
        if False:
            while True:
                i = 10
        pass

    @abstractmethod
    def setExceptionEscape(self, label):
        if False:
            print('Hello World!')
        pass

    @abstractmethod
    def getLoopBreakTarget(self):
        if False:
            print('Hello World!')
        pass

    @abstractmethod
    def setLoopBreakTarget(self, label):
        if False:
            for i in range(10):
                print('nop')
        pass

    @abstractmethod
    def getLoopContinueTarget(self):
        if False:
            i = 10
            return i + 15
        pass

    @abstractmethod
    def setLoopContinueTarget(self, label):
        if False:
            for i in range(10):
                print('nop')
        pass

    @abstractmethod
    def allocateLabel(self, label):
        if False:
            for i in range(10):
                print('nop')
        pass

    @abstractmethod
    def allocateExceptionKeeperVariables(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @abstractmethod
    def getExceptionKeeperVariables(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @abstractmethod
    def setExceptionKeeperVariables(self, keeper_vars):
        if False:
            print('Hello World!')
        pass

    @abstractmethod
    def addExceptionPreserverVariables(self, count):
        if False:
            i = 10
            return i + 15
        pass

    @abstractmethod
    def getTrueBranchTarget(self):
        if False:
            print('Hello World!')
        pass

    @abstractmethod
    def getFalseBranchTarget(self):
        if False:
            while True:
                i = 10
        pass

    @abstractmethod
    def setTrueBranchTarget(self, label):
        if False:
            for i in range(10):
                print('nop')
        pass

    @abstractmethod
    def setFalseBranchTarget(self, label):
        if False:
            i = 10
            return i + 15
        pass

    @abstractmethod
    def getCleanupTempNames(self):
        if False:
            i = 10
            return i + 15
        pass

    @abstractmethod
    def addCleanupTempName(self, tmp_name):
        if False:
            print('Hello World!')
        pass

    @abstractmethod
    def removeCleanupTempName(self, tmp_name):
        if False:
            print('Hello World!')
        pass

    @abstractmethod
    def needsCleanup(self, tmp_name):
        if False:
            return 10
        pass

    @abstractmethod
    def pushCleanupScope(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @abstractmethod
    def popCleanupScope(self):
        if False:
            i = 10
            return i + 15
        pass

class PythonChildContextBase(PythonContextBase):
    __slots__ = ('parent',)

    def __init__(self, parent):
        if False:
            for i in range(10):
                print('nop')
        PythonContextBase.__init__(self)
        self.parent = parent

    def getConstantCode(self, constant, deep_check=False):
        if False:
            for i in range(10):
                print('nop')
        return self.parent.getConstantCode(constant, deep_check=deep_check)

    def getModuleCodeName(self):
        if False:
            i = 10
            return i + 15
        return self.parent.getModuleCodeName()

    def getModuleName(self):
        if False:
            return 10
        return self.parent.getModuleName()

    def addHelperCode(self, key, code):
        if False:
            print('Hello World!')
        return self.parent.addHelperCode(key, code)

    def hasHelperCode(self, key):
        if False:
            print('Hello World!')
        return self.parent.hasHelperCode(key)

    def addDeclaration(self, key, code):
        if False:
            i = 10
            return i + 15
        self.parent.addDeclaration(key, code)

    def pushFrameVariables(self, frame_variables):
        if False:
            print('Hello World!')
        return self.parent.pushFrameVariables(frame_variables)

    def popFrameVariables(self):
        if False:
            while True:
                i = 10
        return self.parent.popFrameVariables()

    def getFrameVariableTypeDescriptions(self):
        if False:
            return 10
        return self.parent.getFrameVariableTypeDescriptions()

    def getFrameVariableTypeDescription(self):
        if False:
            for i in range(10):
                print('nop')
        return self.parent.getFrameVariableTypeDescription()

    def getFrameTypeDescriptionDeclaration(self):
        if False:
            for i in range(10):
                print('nop')
        return self.parent.getFrameTypeDescriptionDeclaration()

    def getFrameVariableCodeNames(self):
        if False:
            i = 10
            return i + 15
        return self.parent.getFrameVariableCodeNames()

    def addFunctionCreationInfo(self, creation_info):
        if False:
            print('Hello World!')
        return self.parent.addFunctionCreationInfo(creation_info)

class FrameDeclarationsMixin(object):
    __slots__ = ()

    def __init__(self):
        if False:
            while True:
                i = 10
        self.frame_variables_stack = ['']
        self.frame_type_descriptions = [()]
        self.frame_variable_types = {}
        self.frames_used = 0
        self.frame_stack = [None]
        self.locals_dict_names = None

    def getFrameHandle(self):
        if False:
            print('Hello World!')
        return self.frame_stack[-1]

    def pushFrameHandle(self, code_identifier, is_light):
        if False:
            for i in range(10):
                print('nop')
        self.frames_used += 1
        if is_light:
            frame_identifier = VariableDeclaration('struct Nuitka_FrameObject *', 'm_frame', None, self.getContextObjectName())
        else:
            frame_handle = code_identifier.replace('codeobj_', 'frame_')
            if self.frames_used > 1:
                frame_handle += '_%d' % self.frames_used
            frame_identifier = self.variable_storage.addVariableDeclarationTop('struct Nuitka_FrameObject *', frame_handle, None)
        self.variable_storage.addVariableDeclarationTop('NUITKA_MAY_BE_UNUSED char const *', 'type_description_%d' % self.frames_used, 'NULL')
        self.frame_stack.append(frame_identifier)
        return frame_identifier

    def popFrameHandle(self):
        if False:
            for i in range(10):
                print('nop')
        result = self.frame_stack[-1]
        del self.frame_stack[-1]
        return result

    def getFramesCount(self):
        if False:
            for i in range(10):
                print('nop')
        return self.frames_used

    def pushFrameVariables(self, frame_variables):
        if False:
            while True:
                i = 10
        'Set current the frame variables.'
        self.frame_variables_stack.append(frame_variables)
        self.frame_type_descriptions.append(set())

    def popFrameVariables(self):
        if False:
            for i in range(10):
                print('nop')
        'End of frame, restore previous ones.'
        del self.frame_variables_stack[-1]
        del self.frame_type_descriptions[-1]

    def setVariableType(self, variable, variable_declaration):
        if False:
            i = 10
            return i + 15
        assert variable.isLocalVariable(), variable
        self.frame_variable_types[variable] = (str(variable_declaration), variable_declaration.getCType().getTypeIndicator())

    def getFrameVariableTypeDescriptions(self):
        if False:
            for i in range(10):
                print('nop')
        return self.frame_type_descriptions[-1]

    def getFrameTypeDescriptionDeclaration(self):
        if False:
            print('Hello World!')
        return self.variable_storage.getVariableDeclarationTop('type_description_%d' % (len(self.frame_stack) - 1))

    def getFrameVariableTypeDescription(self):
        if False:
            i = 10
            return i + 15
        result = ''.join((self.frame_variable_types.get(variable, ('NULL', 'N'))[1] for variable in self.frame_variables_stack[-1]))
        if result:
            self.frame_type_descriptions[-1].add(result)
        return result

    def getFrameVariableCodeNames(self):
        if False:
            while True:
                i = 10
        result = []
        for variable in self.frame_variables_stack[-1]:
            (variable_code_name, variable_code_type) = self.frame_variable_types.get(variable, ('NULL', 'N'))
            if variable_code_type in ('b',):
                result.append('(int)' + variable_code_name)
            else:
                result.append(variable_code_name)
        return result

    def getLocalsDictNames(self):
        if False:
            i = 10
            return i + 15
        return self.locals_dict_names or ()

    def addLocalsDictName(self, locals_dict_name):
        if False:
            while True:
                i = 10
        result = self.variable_storage.getVariableDeclarationTop(locals_dict_name)
        if result is None:
            result = self.variable_storage.addVariableDeclarationTop('PyObject *', locals_dict_name, 'NULL')
        if self.locals_dict_names is None:
            self.locals_dict_names = set()
        self.locals_dict_names.add(result)
        return result

class ReturnReleaseModeMixin(object):
    __slots__ = ()

    def __init__(self):
        if False:
            while True:
                i = 10
        self.return_release_mode = False
        self.return_exit = None

    def setReturnReleaseMode(self, value):
        if False:
            return 10
        result = self.return_release_mode
        self.return_release_mode = value
        return result

    def getReturnReleaseMode(self):
        if False:
            print('Hello World!')
        return self.return_release_mode

    def setReturnTarget(self, label):
        if False:
            while True:
                i = 10
        result = self.return_exit
        self.return_exit = label
        return result

    def getReturnTarget(self):
        if False:
            i = 10
            return i + 15
        return self.return_exit

class ReturnValueNameMixin(object):
    __slots__ = ()

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.return_name = None

    def getReturnValueName(self):
        if False:
            print('Hello World!')
        if self.return_name is None:
            self.return_name = self.allocateTempName('return_value', unique=True)
        return self.return_name

    def setReturnValueName(self, value):
        if False:
            for i in range(10):
                print('nop')
        result = self.return_name
        self.return_name = value
        return result

class PythonModuleContext(FrameDeclarationsMixin, TempMixin, CodeObjectsMixin, ReturnReleaseModeMixin, ReturnValueNameMixin, PythonContextBase):
    __slots__ = ('module', 'name', 'code_name', 'declaration_codes', 'helper_codes', 'frame_handle', 'variable_storage', 'function_table_entries', 'constant_accessor', 'frame_variables_stack', 'frame_type_descriptions', 'frame_variable_types', 'frames_used', 'frame_stack', 'locals_dict_names', 'tmp_names', 'labels', 'exception_escape', 'loop_continue', 'loop_break', 'true_target', 'false_target', 'keeper_variable_count', 'exception_keepers', 'preserver_variable_declaration', 'cleanup_names', 'code_objects', 'return_release_mode', 'return_exit', 'return_name')

    def __init__(self, module, data_filename):
        if False:
            return 10
        PythonContextBase.__init__(self)
        TempMixin.__init__(self)
        CodeObjectsMixin.__init__(self)
        FrameDeclarationsMixin.__init__(self)
        ReturnReleaseModeMixin.__init__(self)
        ReturnValueNameMixin.__init__(self)
        self.module = module
        self.name = module.getFullName()
        self.code_name = module.getCodeName()
        self.declaration_codes = {}
        self.helper_codes = {}
        self.frame_handle = None
        self.variable_storage = VariableStorage(heap_name=None)
        self.function_table_entries = []
        self.constant_accessor = ConstantAccessor(top_level_name='mod_consts', data_filename=data_filename)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '<PythonModuleContext instance for module %s>' % self.name

    def getOwner(self):
        if False:
            return 10
        return self.module

    def getEntryPoint(self):
        if False:
            i = 10
            return i + 15
        return self.module

    def isCompiledPythonModule(self):
        if False:
            return 10
        return True

    def getName(self):
        if False:
            i = 10
            return i + 15
        return self.name

    def mayRaiseException(self):
        if False:
            while True:
                i = 10
        body = self.module.subnode_body
        return body is not None and body.mayRaiseException(BaseException)
    getModuleName = getName

    def getModuleCodeName(self):
        if False:
            while True:
                i = 10
        return self.code_name

    def setFrameGuardMode(self, guard_mode):
        if False:
            for i in range(10):
                print('nop')
        assert guard_mode == 'once'

    def addHelperCode(self, key, code):
        if False:
            for i in range(10):
                print('nop')
        assert key not in self.helper_codes, key
        self.helper_codes[key] = code

    def hasHelperCode(self, key):
        if False:
            while True:
                i = 10
        return key in self.helper_codes

    def getHelperCodes(self):
        if False:
            print('Hello World!')
        return self.helper_codes

    def addDeclaration(self, key, code):
        if False:
            while True:
                i = 10
        assert key not in self.declaration_codes
        self.declaration_codes[key] = code

    def getDeclarations(self):
        if False:
            for i in range(10):
                print('nop')
        return self.declaration_codes

    def mayRecurse(self):
        if False:
            for i in range(10):
                print('nop')
        return False

    def getConstantCode(self, constant, deep_check=False):
        if False:
            return 10
        if deep_check and Options.is_debug:
            assert not isMutable(constant)
        return self.constant_accessor.getConstantCode(constant)

    def getConstantsCount(self):
        if False:
            print('Hello World!')
        return self.constant_accessor.getConstantsCount()

    def addFunctionCreationInfo(self, creation_info):
        if False:
            i = 10
            return i + 15
        self.function_table_entries.append(creation_info)

    def getFunctionCreationInfos(self):
        if False:
            while True:
                i = 10
        result = self.function_table_entries
        del self.function_table_entries
        return result

    @staticmethod
    def getContextObjectName():
        if False:
            for i in range(10):
                print('nop')
        return None

class PythonFunctionContext(FrameDeclarationsMixin, TempMixin, ReturnReleaseModeMixin, ReturnValueNameMixin, PythonChildContextBase):
    __slots__ = ('function', 'frame_handle', 'variable_storage', 'frame_variables_stack', 'frame_type_descriptions', 'frame_variable_types', 'frames_used', 'frame_stack', 'locals_dict_names', 'tmp_names', 'labels', 'exception_escape', 'loop_continue', 'loop_break', 'true_target', 'false_target', 'keeper_variable_count', 'exception_keepers', 'preserver_variable_declaration', 'cleanup_names', 'return_release_mode', 'return_exit', 'return_name')

    def __init__(self, parent, function):
        if False:
            while True:
                i = 10
        PythonChildContextBase.__init__(self, parent=parent)
        TempMixin.__init__(self)
        FrameDeclarationsMixin.__init__(self)
        ReturnReleaseModeMixin.__init__(self)
        ReturnValueNameMixin.__init__(self)
        self.function = function
        self.setExceptionEscape('function_exception_exit')
        self.setReturnTarget('function_return_exit')
        self.frame_handle = None
        self.variable_storage = self._makeVariableStorage()

    def _makeVariableStorage(self):
        if False:
            i = 10
            return i + 15
        return VariableStorage(heap_name=None)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return "<%s for %s '%s'>" % (self.__class__.__name__, 'function' if not self.function.isExpressionClassBodyBase() else 'class', self.function.getName())

    def getFunction(self):
        if False:
            for i in range(10):
                print('nop')
        return self.function

    def getOwner(self):
        if False:
            print('Hello World!')
        return self.function

    def getEntryPoint(self):
        if False:
            i = 10
            return i + 15
        return self.function

    def mayRecurse(self):
        if False:
            while True:
                i = 10
        return True

    def getCodeObjectHandle(self, code_object):
        if False:
            for i in range(10):
                print('nop')
        return self.parent.getCodeObjectHandle(code_object)

class PythonFunctionDirectContext(PythonFunctionContext):
    __slots__ = ()

    @staticmethod
    def getContextObjectName():
        if False:
            while True:
                i = 10
        return None

    @staticmethod
    def isForDirectCall():
        if False:
            for i in range(10):
                print('nop')
        return True

    @staticmethod
    def isForCreatedFunction():
        if False:
            for i in range(10):
                print('nop')
        return False

class PythonGeneratorObjectContext(PythonFunctionContext):
    __slots__ = ()

    def _makeVariableStorage(self):
        if False:
            return 10
        return VariableStorage(heap_name='%s_heap' % self.getContextObjectName())

    @staticmethod
    def isForDirectCall():
        if False:
            while True:
                i = 10
        return False

    @staticmethod
    def isForCreatedFunction():
        if False:
            print('Hello World!')
        return False

    @staticmethod
    def getContextObjectName():
        if False:
            i = 10
            return i + 15
        return 'generator'

    def getGeneratorReturnValueName(self):
        if False:
            return 10
        if python_version >= 768:
            return self.allocateTempName('return_value', 'PyObject *', unique=True)
        else:
            return self.allocateTempName('generator_return', 'bool', unique=True)

class PythonCoroutineObjectContext(PythonGeneratorObjectContext):
    __slots__ = ()

    @staticmethod
    def getContextObjectName():
        if False:
            print('Hello World!')
        return 'coroutine'

class PythonAsyncgenObjectContext(PythonGeneratorObjectContext):
    __slots__ = ()

    @staticmethod
    def getContextObjectName():
        if False:
            for i in range(10):
                print('nop')
        return 'asyncgen'

class PythonFunctionCreatedContext(PythonFunctionContext):
    __slots__ = ()

    @staticmethod
    def getContextObjectName():
        if False:
            i = 10
            return i + 15
        return None

    @staticmethod
    def isForDirectCall():
        if False:
            i = 10
            return i + 15
        return False

    @staticmethod
    def isForCreatedFunction():
        if False:
            print('Hello World!')
        return True

class PythonFunctionOutlineContext(ReturnReleaseModeMixin, ReturnValueNameMixin, PythonChildContextBase):
    __slots__ = ('outline', 'variable_storage', 'return_release_mode', 'return_exit', 'return_name')

    def __init__(self, parent, outline):
        if False:
            return 10
        PythonChildContextBase.__init__(self, parent=parent)
        ReturnReleaseModeMixin.__init__(self)
        ReturnValueNameMixin.__init__(self)
        self.outline = outline
        self.variable_storage = parent.variable_storage

    def getOwner(self):
        if False:
            for i in range(10):
                print('nop')
        return self.outline

    def getEntryPoint(self):
        if False:
            print('Hello World!')
        return self.outline.getEntryPoint()

    def allocateLabel(self, label):
        if False:
            print('Hello World!')
        return self.parent.allocateLabel(label)

    def allocateTempName(self, base_name, type_name='PyObject *', unique=False):
        if False:
            print('Hello World!')
        return self.parent.allocateTempName(base_name, type_name, unique)

    def skipTempName(self, base_name):
        if False:
            return 10
        return self.parent.skipTempName(base_name)

    def hasTempName(self, base_name):
        if False:
            print('Hello World!')
        return self.parent.hasTempName(base_name)

    def getCleanupTempNames(self):
        if False:
            print('Hello World!')
        return self.parent.getCleanupTempNames()

    def addCleanupTempName(self, tmp_name):
        if False:
            for i in range(10):
                print('nop')
        self.parent.addCleanupTempName(tmp_name)

    def transferCleanupTempName(self, tmp_source, tmp_dest):
        if False:
            print('Hello World!')
        self.parent.transferCleanupTempName(tmp_source, tmp_dest)

    def removeCleanupTempName(self, tmp_name):
        if False:
            i = 10
            return i + 15
        self.parent.removeCleanupTempName(tmp_name)

    def needsCleanup(self, tmp_name):
        if False:
            return 10
        return self.parent.needsCleanup(tmp_name)

    def pushCleanupScope(self):
        if False:
            i = 10
            return i + 15
        return self.parent.pushCleanupScope()

    def popCleanupScope(self):
        if False:
            for i in range(10):
                print('nop')
        self.parent.popCleanupScope()

    def getCodeObjectHandle(self, code_object):
        if False:
            print('Hello World!')
        return self.parent.getCodeObjectHandle(code_object)

    def getExceptionEscape(self):
        if False:
            return 10
        return self.parent.getExceptionEscape()

    def setExceptionEscape(self, label):
        if False:
            for i in range(10):
                print('nop')
        return self.parent.setExceptionEscape(label)

    def getLoopBreakTarget(self):
        if False:
            for i in range(10):
                print('nop')
        return self.parent.getLoopBreakTarget()

    def setLoopBreakTarget(self, label):
        if False:
            print('Hello World!')
        return self.parent.setLoopBreakTarget(label)

    def getLoopContinueTarget(self):
        if False:
            print('Hello World!')
        return self.parent.getLoopContinueTarget()

    def setLoopContinueTarget(self, label):
        if False:
            return 10
        return self.parent.setLoopContinueTarget(label)

    def getTrueBranchTarget(self):
        if False:
            for i in range(10):
                print('nop')
        return self.parent.getTrueBranchTarget()

    def getFalseBranchTarget(self):
        if False:
            while True:
                i = 10
        return self.parent.getFalseBranchTarget()

    def setTrueBranchTarget(self, label):
        if False:
            print('Hello World!')
        self.parent.setTrueBranchTarget(label)

    def setFalseBranchTarget(self, label):
        if False:
            print('Hello World!')
        self.parent.setFalseBranchTarget(label)

    def getFrameHandle(self):
        if False:
            while True:
                i = 10
        return self.parent.getFrameHandle()

    def pushFrameHandle(self, code_identifier, is_light):
        if False:
            print('Hello World!')
        return self.parent.pushFrameHandle(code_identifier, is_light)

    def popFrameHandle(self):
        if False:
            print('Hello World!')
        return self.parent.popFrameHandle()

    def getExceptionKeeperVariables(self):
        if False:
            print('Hello World!')
        return self.parent.getExceptionKeeperVariables()

    def setExceptionKeeperVariables(self, keeper_vars):
        if False:
            while True:
                i = 10
        return self.parent.setExceptionKeeperVariables(keeper_vars)

    def setVariableType(self, variable, variable_declaration):
        if False:
            return 10
        self.parent.setVariableType(variable, variable_declaration)

    def getIntResName(self):
        if False:
            while True:
                i = 10
        return self.parent.getIntResName()

    def getBoolResName(self):
        if False:
            while True:
                i = 10
        return self.parent.getBoolResName()

    def allocateExceptionKeeperVariables(self):
        if False:
            print('Hello World!')
        return self.parent.allocateExceptionKeeperVariables()

    def isForDirectCall(self):
        if False:
            for i in range(10):
                print('nop')
        return self.parent.isForDirectCall()

    def mayRecurse(self):
        if False:
            i = 10
            return i + 15
        return self.parent.mayRecurse()

    def getLocalsDictNames(self):
        if False:
            for i in range(10):
                print('nop')
        return self.parent.getLocalsDictNames()

    def addLocalsDictName(self, locals_dict_name):
        if False:
            for i in range(10):
                print('nop')
        return self.parent.addLocalsDictName(locals_dict_name)

    def addExceptionPreserverVariables(self, count):
        if False:
            while True:
                i = 10
        return self.parent.addExceptionPreserverVariables(count)

    def getContextObjectName(self):
        if False:
            while True:
                i = 10
        return self.parent.getContextObjectName()