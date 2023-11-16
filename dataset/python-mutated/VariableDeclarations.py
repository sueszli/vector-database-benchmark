""" Variable declarations
Holds the information necessary to make C code declarations related to a variable.

"""
from contextlib import contextmanager
from .c_types.CTypeBooleans import CTypeBool
from .c_types.CTypeCFloats import CTypeCFloat
from .c_types.CTypeCLongs import CTypeCLong, CTypeCLongDigit
from .c_types.CTypeModuleDictVariables import CTypeModuleDictVariable
from .c_types.CTypeNuitkaBooleans import CTypeNuitkaBoolEnum
from .c_types.CTypeNuitkaInts import CTypeNuitkaIntOrLongStruct
from .c_types.CTypeNuitkaVoids import CTypeNuitkaVoidEnum
from .c_types.CTypePyObjectPointers import CTypeCellObject, CTypePyObjectPtr, CTypePyObjectPtrPtr

class VariableDeclaration(object):
    __slots__ = ('c_type', 'code_name', 'init_value', 'heap_name', 'maybe_unused')

    def __init__(self, c_type, code_name, init_value, heap_name):
        if False:
            return 10
        if c_type.startswith('NUITKA_MAY_BE_UNUSED'):
            self.c_type = c_type[21:]
            self.maybe_unused = True
        else:
            self.c_type = c_type
            self.maybe_unused = False
        self.code_name = code_name
        self.init_value = init_value
        self.heap_name = heap_name

    def makeCFunctionLevelDeclaration(self):
        if False:
            i = 10
            return i + 15
        pos = self.c_type.find('[')
        if pos != -1:
            lead_c_type = self.c_type[:pos]
            suffix_c_type = self.c_type[pos:]
        else:
            lead_c_type = self.c_type
            suffix_c_type = ''
        return '%s%s%s%s%s%s;' % ('NUITKA_MAY_BE_UNUSED ' if self.maybe_unused else '', lead_c_type, ' ' if lead_c_type[-1] != '*' else '', self.code_name, '' if self.init_value is None else ' = %s' % self.init_value, suffix_c_type)

    def makeCStructDeclaration(self):
        if False:
            return 10
        c_type = self.c_type
        if '[' in c_type:
            array_decl = c_type[c_type.find('['):]
            c_type = c_type[:c_type.find('[')]
        else:
            array_decl = ''
        return '%s%s%s%s;' % (c_type, ' ' if self.c_type[-1] != '*' else '', self.code_name, array_decl)

    def makeCStructInit(self):
        if False:
            for i in range(10):
                print('nop')
        if self.init_value is None:
            return None
        assert self.heap_name, repr(self)
        return '%s%s = %s;' % (self.heap_name + '->' if self.heap_name is not None else '', self.code_name, self.init_value)

    def getCType(self):
        if False:
            for i in range(10):
                print('nop')
        c_type = self.c_type
        if c_type == 'PyObject *':
            return CTypePyObjectPtr
        elif c_type == 'struct Nuitka_CellObject *':
            return CTypeCellObject
        elif c_type == 'PyObject **':
            return CTypePyObjectPtrPtr
        elif c_type == 'nuitka_bool':
            return CTypeNuitkaBoolEnum
        elif c_type == 'bool':
            return CTypeBool
        elif c_type == 'nuitka_ilong':
            return CTypeNuitkaIntOrLongStruct
        elif c_type == 'module_var':
            return CTypeModuleDictVariable
        elif c_type == 'nuitka_void':
            return CTypeNuitkaVoidEnum
        elif c_type == 'long':
            return CTypeCLong
        elif c_type == 'nuitka_digit':
            return CTypeCLongDigit
        elif c_type == 'double':
            return CTypeCFloat
        assert False, c_type

    def __str__(self):
        if False:
            while True:
                i = 10
        if self.heap_name:
            return '%s->%s' % (self.heap_name, self.code_name)
        else:
            return self.code_name

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '<VariableDeclaration %s %s = %r>' % (self.c_type, self.code_name, self.init_value)

class VariableStorage(object):
    __slots__ = ('heap_name', 'variable_declarations_heap', 'variable_declarations_main', 'variable_declarations_closure', 'variable_declarations_locals', 'exception_variable_declarations')

    def __init__(self, heap_name):
        if False:
            print('Hello World!')
        self.heap_name = heap_name
        self.variable_declarations_heap = []
        self.variable_declarations_main = []
        self.variable_declarations_closure = []
        self.variable_declarations_locals = []
        self.exception_variable_declarations = None

    @contextmanager
    def withLocalStorage(self):
        if False:
            for i in range(10):
                print('nop')
        'Local storage for only just during context usage.\n\n        This is for automatic removal of that scope. These are supposed\n        to be nestable eventually.\n\n        '
        self.variable_declarations_locals.append([])
        yield
        self.variable_declarations_locals.pop()

    def getVariableDeclarationTop(self, code_name):
        if False:
            while True:
                i = 10
        for variable_declaration in self.variable_declarations_main:
            if variable_declaration.code_name == code_name:
                return variable_declaration
        for variable_declaration in self.variable_declarations_heap:
            if variable_declaration.code_name == code_name:
                return variable_declaration
        return None

    def getVariableDeclarationClosure(self, closure_index):
        if False:
            for i in range(10):
                print('nop')
        return self.variable_declarations_closure[closure_index]

    def addFrameCacheDeclaration(self, frame_identifier):
        if False:
            return 10
        return self.addVariableDeclarationFunction('static struct Nuitka_FrameObject *', 'cache_%s' % frame_identifier, 'NULL')

    def makeCStructLevelDeclarations(self):
        if False:
            for i in range(10):
                print('nop')
        return [variable_declaration.makeCStructDeclaration() for variable_declaration in self.variable_declarations_heap]

    def makeCStructInits(self):
        if False:
            for i in range(10):
                print('nop')
        return [variable_declaration.makeCStructInit() for variable_declaration in self.variable_declarations_heap if variable_declaration.init_value is not None]

    def getExceptionVariableDescriptions(self):
        if False:
            return 10
        if self.exception_variable_declarations is None:
            self.exception_variable_declarations = (self.addVariableDeclarationTop('PyObject *', 'exception_type', 'NULL'), self.addVariableDeclarationTop('PyObject *', 'exception_value', 'NULL'), self.addVariableDeclarationTop('PyTracebackObject *', 'exception_tb', 'NULL'), self.addVariableDeclarationTop('NUITKA_MAY_BE_UNUSED int', 'exception_lineno', '0'))
        return self.exception_variable_declarations

    def addVariableDeclarationLocal(self, c_type, code_name):
        if False:
            for i in range(10):
                print('nop')
        result = VariableDeclaration(c_type, code_name, None, None)
        self.variable_declarations_locals[-1].append(result)
        return result

    def addVariableDeclarationClosure(self, c_type, code_name):
        if False:
            i = 10
            return i + 15
        result = VariableDeclaration(c_type, code_name, None, None)
        self.variable_declarations_closure.append(result)
        return result

    def addVariableDeclarationFunction(self, c_type, code_name, init_value):
        if False:
            i = 10
            return i + 15
        result = VariableDeclaration(c_type, code_name, init_value, None)
        self.variable_declarations_main.append(result)
        return result

    def addVariableDeclarationTop(self, c_type, code_name, init_value):
        if False:
            print('Hello World!')
        result = VariableDeclaration(c_type, code_name, init_value, self.heap_name)
        if self.heap_name is not None:
            self.variable_declarations_heap.append(result)
        else:
            self.variable_declarations_main.append(result)
        return result

    def makeCLocalDeclarations(self):
        if False:
            return 10
        return [variable_declaration.makeCFunctionLevelDeclaration() for variable_declaration in self.variable_declarations_locals[-1]]

    def makeCFunctionLevelDeclarations(self):
        if False:
            for i in range(10):
                print('nop')
        return [variable_declaration.makeCFunctionLevelDeclaration() for variable_declaration in self.variable_declarations_main]

    def getLocalPreservationDeclarations(self):
        if False:
            while True:
                i = 10
        result = []
        for variable_declarations_local in self.variable_declarations_locals:
            result.extend(variable_declarations_local)
        return result