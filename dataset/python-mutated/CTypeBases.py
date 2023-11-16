""" Base class for all C types.

Defines the interface to use by code generation on C types. Different
types then have to overload the class methods.
"""
type_indicators = {'PyObject *': 'o', 'PyObject **': 'O', 'struct Nuitka_CellObject *': 'c', 'nuitka_bool': 'b', 'nuitka_ilong': 'L'}

class CTypeBase(object):
    c_type = None

    @classmethod
    def getTypeIndicator(cls):
        if False:
            i = 10
            return i + 15
        return type_indicators[cls.c_type]

    @classmethod
    def getInitValue(cls, init_from):
        if False:
            return 10
        'Convert to init value for the type.'
        assert False, cls.c_type

    @classmethod
    def getInitTestConditionCode(cls, value_name, inverted):
        if False:
            return 10
        'Get code to test for uninitialized.'
        assert False, cls.c_type

    @classmethod
    def emitVariableAssignCode(cls, value_name, needs_release, tmp_name, ref_count, inplace, emit, context):
        if False:
            while True:
                i = 10
        'Get code to assign local variable.'
        assert False, cls.c_type

    @classmethod
    def getDeleteObjectCode(cls, to_name, value_name, needs_check, tolerant, emit, context):
        if False:
            return 10
        'Get code to delete (del) local variable.'
        assert False, cls.c_type

    @classmethod
    def getVariableArgReferencePassingCode(cls, variable_code_name):
        if False:
            for i in range(10):
                print('nop')
        'Get code to pass variable as reference argument.'
        assert False, cls.c_type

    @classmethod
    def getVariableArgDeclarationCode(cls, variable_code_name):
        if False:
            return 10
        'Get variable declaration code with given name.'
        assert False, cls.c_type

    @classmethod
    def getCellObjectAssignmentCode(cls, target_cell_code, variable_code_name, emit):
        if False:
            for i in range(10):
                print('nop')
        'Get assignment code to given cell object from object.'
        assert False, cls.c_type

    @classmethod
    def emitAssignmentCodeFromBoolCondition(cls, to_name, condition, emit):
        if False:
            i = 10
            return i + 15
        'Get the assignment code from C boolean condition.'
        assert False, cls.c_type

    @classmethod
    def emitAssignmentCodeToNuitkaIntOrLong(cls, to_name, value_name, needs_check, emit, context):
        if False:
            i = 10
            return i + 15
        'Get the assignment code to int or long type.'
        assert False, to_name

    @classmethod
    def getReleaseCode(cls, value_name, needs_check, emit):
        if False:
            print('Hello World!')
        'Get release code for given object.'
        assert False, cls.c_type

    @classmethod
    def emitReinitCode(cls, value_name, emit):
        if False:
            for i in range(10):
                print('nop')
        'Get release code for given object.'
        assert False, cls.c_type

    @classmethod
    def getTakeReferenceCode(cls, value_name, emit):
        if False:
            return 10
        'Take reference code for given object.'
        assert False, cls.c_type

    @classmethod
    def emitTruthCheckCode(cls, to_name, value_name, emit):
        if False:
            return 10
        'Check the truth of a value and indicate exception to an int.'
        assert to_name.c_type == 'int', to_name
        emit('%s = %s ? 1 : 0;' % (to_name, cls.getTruthCheckCode(value_name)))

    @classmethod
    def emitValueAssertionCode(cls, value_name, emit):
        if False:
            for i in range(10):
                print('nop')
        'Assert that the value is not unassigned.'
        assert False, cls.c_type

    @classmethod
    def emitReleaseAssertionCode(cls, value_name, emit):
        if False:
            i = 10
            return i + 15
        'Assert that the container of the value is not released already of unassigned.'
        cls.emitValueAssertionCode(value_name, emit)

    @classmethod
    def emitAssignConversionCode(cls, to_name, value_name, needs_check, emit, context):
        if False:
            i = 10
            return i + 15
        assert False, cls.c_type

class CTypeNotReferenceCountedMixin(object):
    """Mixin for C types, that have no reference counting mechanism."""

    @classmethod
    def getReleaseCode(cls, value_name, needs_check, emit):
        if False:
            while True:
                i = 10
        if not needs_check:
            cls.emitValueAssertionCode(value_name, emit=emit)

    @classmethod
    def getTakeReferenceCode(cls, value_name, emit):
        if False:
            while True:
                i = 10
        pass

    @classmethod
    def emitReleaseAssertionCode(cls, value_name, emit):
        if False:
            print('Hello World!')
        pass