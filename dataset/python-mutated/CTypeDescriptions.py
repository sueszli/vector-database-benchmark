""" C type descriptions. For specific or general C types, provide code generation help.

"""
import math
from abc import abstractmethod
from nuitka.__past__ import long
from nuitka.code_generation.Namify import namifyConstant
from nuitka.utils.SlotMetaClasses import getMetaClassBase

class TypeDescBase(getMetaClassBase('Type', require_slots=False)):
    type_name = None
    type_desc = None
    type_decl = None
    python_requirement = None

    def __init__(self):
        if False:
            print('Hello World!')
        assert self.type_name
        assert self.type_desc
        assert self.type_decl

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '<%s %s %s>' % (self.__class__.__name__, self.type_name, self.type_desc)

    @classmethod
    def getHelperCodeName(cls):
        if False:
            while True:
                i = 10
        return cls.type_name.upper()

    @classmethod
    def getTypeName2(cls):
        if False:
            print('Hello World!')
        return cls.type_name

    @classmethod
    def getTypeName3(cls):
        if False:
            for i in range(10):
                print('nop')
        return cls.type_name

    @classmethod
    def getVariableDecl(cls, variable_name):
        if False:
            return 10
        if cls.type_decl.endswith('*'):
            return cls.type_decl + variable_name
        else:
            return cls.type_decl + ' ' + variable_name

    @classmethod
    def getTypeDecl(cls):
        if False:
            print('Hello World!')
        return cls.type_decl

    @classmethod
    def getCheckValueCode(cls, operand):
        if False:
            i = 10
            return i + 15
        return 'CHECK_OBJECT(%s);' % operand

    @classmethod
    def getTypeValueExpression(cls, operand):
        if False:
            for i in range(10):
                print('nop')
        return 'Py_TYPE(%s)' % operand

    def getTypeNameExpression(self, type_name):
        if False:
            i = 10
            return i + 15
        if self is object_desc:
            return '%s->tp_name' % type_name
        if self.getTypeName2() == self.getTypeName3():
            return '"%s"' % self.getTypeName2()
        if self.python_requirement == 'PYTHON_VERSION < 0x300':
            return '"%s"' % self.getTypeName2()
        elif self.python_requirement == 'PYTHON_VERSION >= 0x300':
            return '"%s"' % self.getTypeName3()
        elif self.python_requirement is None:
            return '(PYTHON_VERSION < 0x300 ? "%s" : "%s")' % (self.getTypeName2(), self.getTypeName3())
        else:
            assert False, self.python_requirement

    def getTypeValueVariableExpression(self, type_name):
        if False:
            i = 10
            return i + 15
        if self is object_desc:
            return type_name
        else:
            return self.getTypeValueExpression(None)

    @abstractmethod
    def getNewStyleNumberTypeCheckExpression(self, operand):
        if False:
            while True:
                i = 10
        pass

    @staticmethod
    def needsIndexConversion():
        if False:
            return 10
        return True

    def isKnownToNotCoerce(self, right):
        if False:
            print('Hello World!')
        if right is self and right is not object_desc:
            return True
        if self in (int_desc, long_desc, float_desc):
            if right in (str_desc, unicode_desc, tuple_desc, list_desc, set_desc, dict_desc, bytes_desc):
                return True
        if self.getNewStyleNumberTypeCheckExpression('dummy') == '1' and right.getNewStyleNumberTypeCheckExpression('dummy') == '1':
            return True
        if self is not object_desc:
            return not self.hasSlot('nb_coerce')
        else:
            return False

    def getMostSpecificType(self, right):
        if False:
            while True:
                i = 10
        if self is not object_desc:
            return self
        else:
            return right

    def getLeastSpecificType(self, right):
        if False:
            return 10
        if self is object_desc:
            return self
        else:
            return right

    @classmethod
    def hasOneOrBothType(cls, right, type_name):
        if False:
            return 10
        if type_name not in (cls.type_name, right.type_name):
            return False
        if cls.type_name == right.type_name:
            return True
        if 'object' not in (cls.type_name, right.type_name):
            return False
        return True

    @classmethod
    def mayBothHaveType(cls, right, type_name):
        if False:
            return 10
        return cls.type_name in (type_name, 'object') and right.type_name in (type_name, 'object')

    @classmethod
    def getTypeCheckExactExpression(cls, operand):
        if False:
            for i in range(10):
                print('nop')
        if cls.type_name == 'str':
            return 'PyStr_CheckExact(%s)' % operand
        elif cls.type_name == 'dict':
            return 'PyDict_CheckExact(%s)' % operand
        else:
            assert False, cls

    @classmethod
    def getIntCheckExpression(cls, operand):
        if False:
            while True:
                i = 10
        if cls.type_name == 'int':
            return '1'
        elif cls.type_name == 'object':
            return 'PyInt_CheckExact(%s)' % operand
        else:
            return '0'

    @classmethod
    def getLongCheckExpression(cls, operand):
        if False:
            print('Hello World!')
        if cls.type_name == 'long':
            return '1'
        elif cls.type_name == 'object':
            return 'PyLong_CheckExact(%s)' % operand
        else:
            return '0'

    @classmethod
    def getStringCheckExpression(cls, operand):
        if False:
            print('Hello World!')
        if cls.type_name == 'str':
            return '1'
        elif cls.type_name == 'object':
            return 'PyString_CheckExact(%s)' % operand
        else:
            return '0'

    @classmethod
    def getBytesCheckExpression(cls, operand):
        if False:
            print('Hello World!')
        if cls.type_name == 'bytes':
            return '1'
        elif cls.type_name == 'object':
            return 'PyBytes_CheckExact(%s)' % operand
        else:
            return '0'

    @classmethod
    def getUnicodeCheckExpression(cls, operand):
        if False:
            print('Hello World!')
        if cls.type_name == 'unicode':
            return '1'
        elif cls.type_name == 'object':
            return 'PyUnicode_CheckExact(%s)' % operand
        else:
            return '0'

    @classmethod
    def getFloatCheckExpression(cls, operand):
        if False:
            i = 10
            return i + 15
        if cls.type_name == 'float':
            return '1'
        elif cls.type_name == 'object':
            return 'PyFloat_CheckExact(%s)' % operand
        else:
            return '0'

    @classmethod
    def getListCheckExpression(cls, operand):
        if False:
            print('Hello World!')
        if cls.type_name == 'list':
            return '1'
        elif cls.type_name == 'object':
            return 'PyList_CheckExact(%s)' % operand
        else:
            return '0'

    def getSequenceCheckExpression(self, operand, right):
        if False:
            while True:
                i = 10
        if self.type_name == 'dict':
            return '0'
        elif self.type_name == 'object':
            if right.type_name == 'tuple':
                return '(PyTuple_CheckExact(%s) || PySequence_Check(%s))' % (operand, operand)
            else:
                return 'PySequence_Check(%s)' % operand
        elif self.hasSlot('sq_item'):
            return '1'
        else:
            return '0'

    def getInstanceCheckCode(self, operand):
        if False:
            print('Hello World!')
        if self.type_name == 'object':
            return 'PyInstance_Check(%s)' % operand
        else:
            return '0'

    def getIndexCheckExpression(self, operand):
        if False:
            return 10
        if self.hasSlot('nb_index'):
            return '1'
        elif self.type_name == 'object':
            return 'Nuitka_Index_Check(%s)' % operand
        else:
            return '0'

    def getSaneTypeCheckCode(self, operand):
        if False:
            print('Hello World!')
        if self.type_name == 'object':
            return 'IS_SANE_TYPE(Py_TYPE(%s))' % operand
        elif self in (str_desc, int_desc, long_desc, list_desc, tuple_desc):
            return '1'
        elif self in (float_desc,):
            return '0'
        else:
            assert False, self
            return '0'

    def getTypeIdenticalCheckExpression(self, other, type1, type2):
        if False:
            for i in range(10):
                print('nop')
        if self is object_desc or other is object_desc:
            return '%s == %s' % (self.getTypeValueExpression(None) if self is not object_desc else type1, other.getTypeValueExpression(None) if other is not object_desc else type2)
        elif self is other:
            return '1'
        else:
            return '0'

    def getTypeNonIdenticalCheckExpression(self, other, type1, type2):
        if False:
            print('Hello World!')
        if self is object_desc or other is object_desc:
            return '%s != %s' % (self.getTypeValueExpression(None) if self is not object_desc else type1, other.getTypeValueExpression(None) if other is not object_desc else type2)
        elif self is other:
            return '0'
        else:
            return '1'

    def getTypeSubTypeCheckExpression(self, other, type2, type1):
        if False:
            print('Hello World!')
        return 'Nuitka_Type_IsSubtype(%s, %s)' % (other.getTypeValueExpression(None) if other is not object_desc else type2, self.getTypeValueExpression(None) if self is not object_desc else type1)

    def getRealSubTypeCheckCode(self, other, type2, type1):
        if False:
            while True:
                i = 10
        if other is object_desc:
            return 'Nuitka_Type_IsSubtype(%s, %s)' % (type2, self.getTypeValueExpression(None) if self is not object_desc else type1)
        else:
            return 0

    @abstractmethod
    def hasSlot(self, slot):
        if False:
            i = 10
            return i + 15
        pass

    @staticmethod
    def hasPreferredSlot(right, slot):
        if False:
            return 10
        return False

    def _getSlotValueExpression(self, operand, slot):
        if False:
            for i in range(10):
                print('nop')
        if slot.startswith('nb_'):
            return '(%s) ? %s : NULL' % (operand + '->tp_as_number != NULL && ' + self.getNewStyleNumberTypeCheckExpression(operand), operand + '->tp_as_number->' + slot)
        elif slot.startswith('sq_'):
            return '%s ? %s : NULL' % (operand + '->tp_as_sequence' + ' != NULL', operand + '->tp_as_sequence->' + slot)
        elif slot == 'tp_richcompare':
            assert self is object_desc, self
            return 'RICHCOMPARE(%s)' % operand
        elif slot == 'tp_compare':
            return operand + '->tp_compare'
        else:
            assert False, slot

    @staticmethod
    def getSlotType(slot):
        if False:
            for i in range(10):
                print('nop')
        if slot in ('nb_power', 'nb_inplace_power'):
            return 'ternaryfunc'
        elif slot in ('sq_repeat', 'sq_inplace_repeat'):
            return 'ssizeargfunc'
        else:
            return 'binaryfunc'

    @staticmethod
    def getSlotCallExpression(nb_slot, slot_var, operand1, operand2):
        if False:
            while True:
                i = 10
        if nb_slot in ('nb_power', 'nb_inplace_power'):
            return '%s(%s, %s, Py_None)' % (slot_var, operand1, operand2)
        else:
            return '%s(%s, %s)' % (slot_var, operand1, operand2)

    def getSlotValueExpression(self, operand, slot):
        if False:
            for i in range(10):
                print('nop')
        assert 'inplace_' not in slot or not self.hasSlot(slot) or self in (set_desc, list_desc), self.hasSlot
        if not self.hasSlot(slot):
            return 'NULL'
        return self._getSlotValueExpression(operand, slot)

    def getSlotValueCheckExpression(self, operand, slot):
        if False:
            while True:
                i = 10
        return 'true' if self.hasSlot(slot) else 'false'

    @abstractmethod
    def getNoSequenceSlotAccessTestCode(self, type_name):
        if False:
            print('Hello World!')
        pass

    @staticmethod
    def getOperationErrorMessageName(operator):
        if False:
            print('Hello World!')
        if operator == '%':
            return '%%'
        elif operator == '**':
            return '** or pow()'
        elif operator == 'divmod':
            return 'divmod()'
        else:
            return operator

    def getReturnUnorderableTypeErrorCode(self, operator, left, right, operand1, operand2):
        if False:
            while True:
                i = 10
        args = []
        if left is object_desc:
            args.append('%s->tp_name' % operand1)
        if right is object_desc:
            args.append('%s->tp_name' % operand2)
        if args:
            args = ', ' + ', '.join(args)
        else:
            args = ''
        if left.getTypeName2() != left.getTypeName3() or right.getTypeName2() != right.getTypeName3():
            return '#if PYTHON_VERSION < 0x300\nPyErr_Format(PyExc_TypeError, "unorderable types: %(left_type2)s() %(operator)s %(right_type2)s()"%(args)s);\n#elif PYTHON_VERSION < 0x360\nPyErr_Format(PyExc_TypeError, "unorderable types: %(left_type3)s() %(operator)s %(right_type3)s()"%(args)s);\n#else\nPyErr_Format(PyExc_TypeError, "\'%(operator)s\' not supported between instances of \'%(left_type3)s\' and \'%(right_type3)s\'"%(args)s);\n#endif\nreturn %(return_value)s;' % {'operator': operator, 'left_type2': '%s' if left is object_desc else left.getTypeName2(), 'right_type2': '%s' if right is object_desc else right.getTypeName2(), 'left_type3': '%s' if left is object_desc else left.getTypeName3(), 'right_type3': '%s' if right is object_desc else right.getTypeName3(), 'args': args, 'return_value': self.getExceptionResultIndicatorValue()}
        else:
            return '#if PYTHON_VERSION < 0x360\nPyErr_Format(PyExc_TypeError, "unorderable types: %(left_type)s() %(operator)s %(right_type)s()"%(args)s);\n#else\nPyErr_Format(PyExc_TypeError, "\'%(operator)s\' not supported between instances of \'%(left_type)s\' and \'%(right_type)s\'"%(args)s);\n#endif\nreturn %(return_value)s;' % {'operator': operator, 'left_type': '%s' if left is object_desc else left.getTypeName2(), 'right_type': '%s' if right is object_desc else right.getTypeName2(), 'args': args, 'return_value': self.getExceptionResultIndicatorValue()}

    def hasSameTypeOperationSpecializationCode(self, other, nb_slot, sq_slot):
        if False:
            i = 10
            return i + 15
        candidate = self if self is not object_desc else other
        if candidate is object_desc:
            assert self is object_desc
            assert other is object_desc
            return False
        if sq_slot is not None and (not candidate.hasSlot(nb_slot)) and candidate.hasSlot(sq_slot):
            slot = sq_slot
        else:
            slot = nb_slot
        if slot == 'sq_repeat':
            if candidate in (list_desc, tuple_desc, set_desc, dict_desc, unicode_desc, str_desc, bytes_desc):
                return False
        if slot == 'nb_remainder':
            if candidate in (list_desc, tuple_desc, set_desc, dict_desc):
                return False
        if slot == 'nb_multiply':
            if candidate in (str_desc, bytes_desc, list_desc, tuple_desc, set_desc, dict_desc):
                return False
        if slot == 'nb_add':
            if candidate in (str_desc, bytes_desc, tuple_desc, list_desc, set_desc, dict_desc):
                return False
        if slot in ('nb_and', 'nb_or', 'nb_xor'):
            if candidate in (str_desc, bytes_desc, unicode_desc, list_desc, tuple_desc, dict_desc, float_desc):
                return False
        if slot in ('nb_lshift', 'nb_rshift'):
            if candidate in (str_desc, bytes_desc, unicode_desc, tuple_desc, list_desc, set_desc, dict_desc, float_desc):
                return False
        if slot == 'nb_matrix_multiply':
            return False
        return True

    def hasSimilarTypeSpecializationCode(self, other):
        if False:
            while True:
                i = 10
        return other in related_types.get(self, ())

    def getSameTypeType(self, other):
        if False:
            print('Hello World!')
        if self is object_desc:
            return other
        elif other is object_desc:
            return self
        else:
            return object_desc

    def isSimilarOrSameTypesAsOneOf(self, *others):
        if False:
            return 10
        for other in others:
            assert other is not None
            if self is other or other in related_types.get(self, ()):
                return True
        return False

    def hasTypeSpecializationCode(self, other, nb_slot, sq_slot):
        if False:
            print('Hello World!')
        if self is object_desc and other is object_desc:
            return False
        if self is other:
            return self.hasSameTypeOperationSpecializationCode(other=other, nb_slot=nb_slot, sq_slot=sq_slot)
        return self.hasSimilarTypeSpecializationCode(other=other)

    def getTypeComparisonSpecializationHelper(self, other, op_code, target, operand1, operand2):
        if False:
            for i in range(10):
                print('nop')
        candidate1 = self if self is not object_desc else other
        candidate2 = other if other is not object_desc else self
        if candidate1 is object_desc:
            return ('', None, None, None, None, None)
        if long_desc in (candidate1, candidate2) and int_desc in (candidate1, candidate2):
            if candidate1 == int_desc:
                operand1 = int_desc.getAsLongValueExpression(operand1)
                candidate1 = c_long_desc
            elif candidate2 == int_desc:
                operand2 = int_desc.getAsLongValueExpression(operand1)
                candidate2 = c_long_desc
            else:
                assert False
        if target is n_bool_desc and candidate1 is candidate2 and (candidate1 not in (tuple_desc, list_desc)):
            target = c_bool_desc
        return ('COMPARE_%s_%s_%s_%s' % (op_code, target.getHelperCodeName(), candidate1.getHelperCodeName(), candidate2.getHelperCodeName()), target, candidate1, candidate2, operand1, operand2)

    def getTypeComparisonSpecializationCode(self, other, op_code, target, operand1, operand2):
        if False:
            while True:
                i = 10
        if target is n_bool_desc and self not in (tuple_desc, list_desc) and (other not in (tuple_desc, list_desc)):
            helper_target = c_bool_desc
        else:
            helper_target = target
        (helper_name, _helper_target, _type_desc1, _type_desc2, operand1, operand2) = self.getTypeComparisonSpecializationHelper(other=other, op_code=op_code, target=helper_target, operand1=operand1, operand2=operand2)
        if not helper_name:
            return ''
        assert helper_name != 'COMPARE_GE_NBOOL_INT_INT'
        if helper_target is target:
            return 'return %s(%s, %s);' % (helper_name, operand1, operand2)
        else:
            return 'return %s(%s, %s) ? NUITKA_BOOL_TRUE : NUITKA_BOOL_FALSE;' % (helper_name, operand1, operand2)

    @staticmethod
    def getTakeReferenceStatement(operand):
        if False:
            i = 10
            return i + 15
        return 'Py_INCREF(%s);' % operand

    @classmethod
    def hasReferenceCounting(cls):
        if False:
            for i in range(10):
                print('nop')
        return True

    @classmethod
    def getReturnFromObjectExpressionCode(cls, operand, take_ref=False, check_exception=True):
        if False:
            while True:
                i = 10
        if check_exception and (not (cls.type_name == 'object' and (not take_ref))):
            r = 'if (unlikely(%s == NULL)) { return %s; }\n\n' % (operand, cls.getExceptionResultIndicatorValue())
        else:
            r = ''
        return r + cls._getReturnFromObjectExpressionCode(operand=operand, take_ref=take_ref)

    @classmethod
    def getAssignFromObjectExpressionCode(cls, result, operand, take_ref=False):
        if False:
            return 10
        if cls.type_name == 'object':
            if take_ref:
                return 'Py_INCREF(%s); %s = %s;' % (operand, result, operand)
            else:
                return '%s = %s;' % (result, operand)
        elif take_ref:
            return '%s = %s; ' % (result, cls.getToValueFromObjectExpression(operand))
        else:
            return '%s = %s; Py_DECREF(%s); ' % (result, cls.getToValueFromObjectExpression(operand), operand)

    @classmethod
    def _getReturnFromObjectExpressionCode(cls, operand, take_ref):
        if False:
            for i in range(10):
                print('nop')
        if cls.type_name == 'object':
            if take_ref:
                return 'Py_INCREF(%s); return %s;' % (operand, operand)
            else:
                return 'return %s;' % operand
        elif take_ref:
            return '{ %s r = %s; return r; }' % (cls.getTypeDecl(), cls.getToValueFromObjectExpression(operand))
        else:
            return '{ %s r = %s; Py_DECREF(%s); return r; }' % (cls.getTypeDecl(), cls.getToValueFromObjectExpression(operand), operand)

    @classmethod
    def getReturnFromLongExpressionCode(cls, operand):
        if False:
            while True:
                i = 10
        if cls.type_name == 'object':
            assert False
        elif cls.type_name == 'nbool':
            return 'return %s;' % cls.getToValueFromBoolExpression('%s != 0' % operand)
        else:
            assert False, cls

    @classmethod
    def getAssignFromLongExpressionCode(cls, result, operand):
        if False:
            for i in range(10):
                print('nop')
        if cls.type_name == 'object':
            assert False
        elif cls.type_name == 'int':
            return '%s = PyInt_FromLong(%s);' % (result, operand)
        elif cls.type_name == 'long':
            return '%s = Nuitka_LongFromCLong(%s);' % (result, operand)
        elif cls.type_name == 'nbool':
            return '%s = %s;' % (result, cls.getToValueFromBoolExpression('%s != 0' % operand))
        else:
            assert False, cls

    @classmethod
    def getAssignFromBoolExpressionCode(cls, result, operand, give_ref):
        if False:
            return 10
        if cls.type_name == 'object':
            code = '%s = BOOL_FROM(%s);' % (result, operand)
            if give_ref:
                code += 'Py_INCREF(%s);' % result
            return code
        elif cls.type_name == 'nbool':
            return '%s = %s;' % (result, cls.getToValueFromBoolExpression('%s' % operand))
        else:
            assert False, cls

    @classmethod
    def getReturnFromFloatExpressionCode(cls, operand):
        if False:
            for i in range(10):
                print('nop')
        if cls.type_name == 'object':
            return 'return MAKE_FLOAT_FROM_DOUBLE(%s);' % operand
        elif cls.type_name == 'nbool':
            return 'return %s;' % cls.getToValueFromBoolExpression('%s == 0.0' % operand)
        elif cls.type_name == 'float':
            return 'return %s;' % operand
        else:
            assert False, cls

    @classmethod
    def getAssignFromFloatExpressionCode(cls, result, operand):
        if False:
            i = 10
            return i + 15
        if cls.type_name in ('object', 'int', 'float'):
            return '%s = MAKE_FLOAT_FROM_DOUBLE(%s);' % (result, operand)
        elif cls.type_name == 'nbool':
            return '%s = %s;' % (result, cls.getToValueFromBoolExpression('%s != 0.0' % operand))
        elif cls.type_name == 'float':
            return '%s = %s;' % (result, operand)
        else:
            assert False, cls

    @classmethod
    def getReturnFromFloatConstantCode(cls, value):
        if False:
            i = 10
            return i + 15
        if cls.type_name == 'object':
            const_name = 'const_' + namifyConstant(value)
            return 'Py_INCREF(%(const_name)s); return %(const_name)s;' % {'const_name': const_name}
        elif cls.type_name in ('nbool', 'float'):
            return cls.getReturnFromFloatExpressionCode(value)
        else:
            assert False, cls

    @classmethod
    def getAssignFromFloatConstantCode(cls, result, value):
        if False:
            print('Hello World!')
        if value == 'nan':
            value = float(value)
        if cls.type_name in ('object', 'int'):
            const_name = 'const_' + namifyConstant(value)
            return 'Py_INCREF(%(const_name)s); %(result)s = %(const_name)s;' % {'result': result, 'const_name': const_name}
        elif cls.type_name in ('nbool', 'float'):
            if math.isnan(value):
                value = 'Py_NAN'
            return cls.getAssignFromFloatExpressionCode(result, value)
        else:
            assert False, cls

    @classmethod
    def getReturnFromIntConstantCode(cls, value):
        if False:
            i = 10
            return i + 15
        if cls.type_name == 'object':
            const_name = 'const_' + namifyConstant(value)
            return 'Py_INCREF(%(const_name)s); return %(const_name)s;' % {'const_name': const_name}
        elif cls.type_name in ('nbool', 'float'):
            return cls.getReturnFromLongExpressionCode(value)
        else:
            assert False, cls

    @classmethod
    def getAssignFromIntConstantCode(cls, result, value):
        if False:
            i = 10
            return i + 15
        if cls.type_name in ('object', 'int'):
            const_name = 'const_' + namifyConstant(value)
            return 'Py_INCREF(%(const_name)s); %(result)s = %(const_name)s;' % {'result': result, 'const_name': const_name}
        elif cls.type_name in ('nbool', 'float'):
            return cls.getAssignFromLongExpressionCode(result, value)
        else:
            assert False, (cls, cls.type_name)

    @classmethod
    def getAssignFromLongConstantCode(cls, result, value):
        if False:
            i = 10
            return i + 15
        if cls.type_name in ('object', 'long'):
            if str is bytes:
                value = long(value)
            assert value == 0
            const_name2 = 'const_' + namifyConstant(value)
            const_name3 = 'Nuitka_Long_GetSmallValue(%s)' % value
            assert -5 >= value < 256, value
            return '#if PYTHON_VERSION < 0x300\n%(result)s = %(const_name2)s;\n#else\n%(result)s = %(const_name3)s;\n#endif\nPy_INCREF(%(result)s);' % {'result': result, 'const_name2': const_name2, 'const_name3': const_name3}
        elif cls.type_name in ('nbool', 'float'):
            return cls.getAssignFromLongExpressionCode(result, value)
        else:
            assert False, (cls, cls.type_name)

    @classmethod
    def getAssignConversionCode(cls, result, left, value):
        if False:
            for i in range(10):
                print('nop')

        def _getObjectObject():
            if False:
                for i in range(10):
                    print('nop')
            code = '%s = %s;' % (result, value)
            code += cls.getTakeReferenceStatement(result)
            return code
        if cls is left:
            return _getObjectObject()
        elif cls.type_name in ('object', 'float'):
            if left.type_name in ('int', 'float'):
                return _getObjectObject()
            elif left.type_name == 'clong':
                return cls.getAssignFromLongExpressionCode(result, value)
            else:
                assert False, left.type_name
        elif cls.type_name == 'nbool':
            if left.type_name == 'int':
                return '%s = %s;' % (result, cls.getToValueFromBoolExpression('%s != 0' % left.getAsLongValueExpression(value)))
            elif left.type_name == 'float':
                return '%s = %s;' % (result, cls.getToValueFromBoolExpression('%s != 0.0' % left.getAsDoubleValueExpression(value)))
            else:
                assert False, left.type_name
        else:
            assert False, cls.type_name

class ConcreteTypeBase(TypeDescBase):
    type_decl = 'PyObject *'

    def _getSlotValueExpression(self, operand, slot):
        if False:
            return 10
        if slot.startswith('nb_'):
            return self.getTypeValueExpression(operand)[1:] + '.tp_as_number->' + slot
        elif slot.startswith('sq_'):
            return self.getTypeValueExpression(operand)[1:] + '.tp_as_sequence->' + slot
        elif slot.startswith('tp_'):
            return self.getTypeValueExpression(operand)[1:] + '.' + slot
        else:
            assert False, slot

    def getCheckValueCode(self, operand):
        if False:
            while True:
                i = 10
        return 'CHECK_OBJECT(%(operand)s);\nassert(%(type_name)s_CheckExact(%(operand)s));' % {'operand': operand, 'type_name': self.getTypeValueExpression(operand)[1:].split('_')[0]}

    @abstractmethod
    def getTypeValueExpression(self, operand):
        if False:
            return 10
        pass

    @staticmethod
    def getTakeReferenceStatement(operand):
        if False:
            i = 10
            return i + 15
        return ''

class ConcreteNonSequenceTypeBase(ConcreteTypeBase):
    """Base class for concrete types that are not sequences."""

    def getNoSequenceSlotAccessTestCode(self, type_name):
        if False:
            print('Hello World!')
        'Test if type has no sequence slots at all.'
        return 'true'

class ConcreteSequenceTypeBase(ConcreteTypeBase):
    """Base class for concrete types that are sequences."""

    def getNoSequenceSlotAccessTestCode(self, type_name):
        if False:
            while True:
                i = 10
        'Test if type has no sequence slots at all.'
        return 'false'

class IntDesc(ConcreteNonSequenceTypeBase):
    type_name = 'int'
    type_desc = "Python2 'int'"
    python_requirement = 'PYTHON_VERSION < 0x300'

    @classmethod
    def getTypeValueExpression(cls, operand):
        if False:
            return 10
        return '&PyInt_Type'

    @classmethod
    def getNewStyleNumberTypeCheckExpression(cls, operand):
        if False:
            print('Hello World!')
        return '1'

    def hasSlot(self, slot):
        if False:
            print('Hello World!')
        if slot.startswith('nb_inplace'):
            return False
        elif slot.startswith('nb_'):
            return slot != 'nb_matrix_multiply'
        elif slot.startswith('sq_'):
            return False
        elif slot == 'tp_richcompare':
            return False
        elif slot == 'tp_compare':
            return True
        else:
            assert False

    @staticmethod
    def needsIndexConversion():
        if False:
            for i in range(10):
                print('nop')
        return False

    @staticmethod
    def getAsLongValueExpression(operand):
        if False:
            print('Hello World!')
        return 'PyInt_AS_LONG(%s)' % operand

    @staticmethod
    def getAsObjectValueExpression(operand):
        if False:
            for i in range(10):
                print('nop')
        return operand

    @staticmethod
    def releaseAsObjectValueStatement(operand):
        if False:
            i = 10
            return i + 15
        return ''
int_desc = IntDesc()

class StrDesc(ConcreteSequenceTypeBase):
    type_name = 'str'
    type_desc = "Python2 'str'"
    python_requirement = 'PYTHON_VERSION < 0x300'

    @classmethod
    def getTypeValueExpression(cls, operand):
        if False:
            i = 10
            return i + 15
        return '&PyString_Type'

    @classmethod
    def getNewStyleNumberTypeCheckExpression(cls, operand):
        if False:
            return 10
        return '1'

    def hasSlot(self, slot):
        if False:
            print('Hello World!')
        if slot.startswith('nb_'):
            return slot == 'nb_remainder'
        elif slot.startswith('sq_'):
            return 'ass' not in slot and 'inplace' not in slot
        elif slot == 'tp_richcompare':
            return True
        elif slot == 'tp_compare':
            return False
        else:
            assert False, (self, slot)

    def hasPreferredSlot(self, right, slot):
        if False:
            return 10
        if slot == 'nb_multiply':
            return right in (int_desc, long_desc)
        return False
str_desc = StrDesc()

class UnicodeDesc(ConcreteSequenceTypeBase):
    type_name = 'unicode'
    type_desc = "Python2 'unicode', Python3 'str'"

    @classmethod
    def getTypeName3(cls):
        if False:
            i = 10
            return i + 15
        return 'str'

    @classmethod
    def getTypeValueExpression(cls, operand):
        if False:
            while True:
                i = 10
        return '&PyUnicode_Type'

    @classmethod
    def getCheckValueCode(cls, operand):
        if False:
            print('Hello World!')
        return 'CHECK_OBJECT(%(operand)s);\nassert(PyUnicode_CheckExact(%(operand)s));' % {'operand': operand}

    @classmethod
    def getNewStyleNumberTypeCheckExpression(cls, operand):
        if False:
            i = 10
            return i + 15
        return '1'

    def hasSlot(self, slot):
        if False:
            return 10
        if slot.startswith('nb_'):
            return slot == 'nb_remainder'
        elif slot.startswith('sq_'):
            return 'ass' not in slot and 'inplace' not in slot
        elif slot == 'tp_richcompare':
            return True
        elif slot == 'tp_compare':
            return True
        else:
            assert False, slot

    def hasPreferredSlot(self, right, slot):
        if False:
            return 10
        if slot == 'sq_concat' and right is str_desc:
            return True
        if slot == 'nb_multiply':
            return right in (int_desc, long_desc)
        return False
unicode_desc = UnicodeDesc()

class FloatDesc(ConcreteNonSequenceTypeBase):
    type_name = 'float'
    type_desc = "Python 'float'"

    @classmethod
    def getTypeValueExpression(cls, operand):
        if False:
            while True:
                i = 10
        return '&PyFloat_Type'

    @staticmethod
    def getAsDoubleValueExpression(operand):
        if False:
            while True:
                i = 10
        return 'PyFloat_AS_DOUBLE(%s)' % operand

    def hasSlot(self, slot):
        if False:
            for i in range(10):
                print('nop')
        if slot.startswith('nb_inplace'):
            return False
        elif slot.startswith('nb_'):
            return slot != 'nb_matrix_multiply'
        elif slot.startswith('sq_'):
            return False
        elif slot == 'tp_richcompare':
            return True
        elif slot == 'tp_compare':
            return False
        else:
            assert False, slot

    @classmethod
    def getNewStyleNumberTypeCheckExpression(cls, operand):
        if False:
            for i in range(10):
                print('nop')
        return '1'

    def hasPreferredSlot(self, right, slot):
        if False:
            while True:
                i = 10
        if right in (int_desc, long_desc):
            return True
        return False
float_desc = FloatDesc()

class TupleDesc(ConcreteSequenceTypeBase):
    type_name = 'tuple'
    type_desc = "Python 'tuple'"

    @classmethod
    def getTypeValueExpression(cls, operand):
        if False:
            while True:
                i = 10
        return '&PyTuple_Type'

    def hasSlot(self, slot):
        if False:
            return 10
        if slot.startswith('nb_'):
            return False
        elif slot.startswith('sq_'):
            return 'ass' not in slot and 'inplace' not in slot
        elif slot == 'tp_richcompare':
            return True
        elif slot == 'tp_compare':
            return False
        else:
            assert False, slot

    @classmethod
    def getNewStyleNumberTypeCheckExpression(cls, operand):
        if False:
            return 10
        return '0'

    def hasPreferredSlot(self, right, slot):
        if False:
            for i in range(10):
                print('nop')
        if slot == 'nb_multiply':
            return right in (int_desc, long_desc)
        return False
tuple_desc = TupleDesc()

class ListDesc(ConcreteSequenceTypeBase):
    type_name = 'list'
    type_desc = "Python 'list'"

    @classmethod
    def getTypeValueExpression(cls, operand):
        if False:
            while True:
                i = 10
        return '&PyList_Type'

    def hasSlot(self, slot):
        if False:
            while True:
                i = 10
        if slot.startswith('nb_'):
            return False
        elif slot.startswith('sq_'):
            return True
        elif slot == 'tp_richcompare':
            return True
        elif slot == 'tp_compare':
            return False
        else:
            assert False, slot

    @classmethod
    def getNewStyleNumberTypeCheckExpression(cls, operand):
        if False:
            print('Hello World!')
        return '0'

    def hasPreferredSlot(self, right, slot):
        if False:
            while True:
                i = 10
        if slot == 'nb_multiply':
            return right in (int_desc, long_desc)
        return False
list_desc = ListDesc()

class SetDesc(ConcreteSequenceTypeBase):
    type_name = 'set'
    type_desc = "Python 'set'"

    @classmethod
    def getTypeValueExpression(cls, operand):
        if False:
            print('Hello World!')
        return '&PySet_Type'

    def hasSlot(self, slot):
        if False:
            return 10
        if slot.startswith('nb_inplace_'):
            return slot in ('nb_inplace_subtract', 'nb_inplace_and', 'nb_inplace_or', 'nb_inplace_xor')
        elif slot.startswith('nb_'):
            return slot in ('nb_subtract', 'nb_and', 'nb_or', 'nb_xor')
        elif slot.startswith('sq_'):
            return True
        else:
            assert False, slot

    @classmethod
    def getNewStyleNumberTypeCheckExpression(cls, operand):
        if False:
            while True:
                i = 10
        return '0'
set_desc = SetDesc()

class DictDesc(ConcreteSequenceTypeBase):
    type_name = 'dict'
    type_desc = "Python 'dict'"

    @classmethod
    def getTypeValueExpression(cls, operand):
        if False:
            print('Hello World!')
        return '&PyDict_Type'

    def hasSlot(self, slot):
        if False:
            return 10
        if slot.startswith('nb_'):
            return False
        elif slot.startswith('sq_'):
            return True
        else:
            assert False, slot

    @classmethod
    def getNewStyleNumberTypeCheckExpression(cls, operand):
        if False:
            print('Hello World!')
        return '0'
dict_desc = DictDesc()

class BytesDesc(ConcreteSequenceTypeBase):
    type_name = 'bytes'
    type_desc = "Python3 'bytes'"
    python_requirement = 'PYTHON_VERSION >= 0x300'

    @classmethod
    def getTypeValueExpression(cls, operand):
        if False:
            i = 10
            return i + 15
        return '&PyBytes_Type'

    def hasSlot(self, slot):
        if False:
            print('Hello World!')
        if slot.startswith('nb_'):
            return slot == 'nb_remainder'
        elif slot.startswith('sq_'):
            return 'ass' not in slot and slot != 'sq_slice' and ('inplace' not in slot)
        elif slot == 'tp_richcompare':
            return True
        elif slot == 'tp_compare':
            return False
        else:
            assert False, slot

    @classmethod
    def getNewStyleNumberTypeCheckExpression(cls, operand):
        if False:
            print('Hello World!')
        return '0'

    def hasPreferredSlot(self, right, slot):
        if False:
            print('Hello World!')
        if slot == 'nb_multiply':
            return right in (int_desc, long_desc)
        return False
bytes_desc = BytesDesc()

class LongDesc(ConcreteNonSequenceTypeBase):
    type_name = 'long'
    type_desc = "Python2 'long', Python3 'int'"

    @classmethod
    def getTypeName3(cls):
        if False:
            for i in range(10):
                print('nop')
        return 'int'

    @classmethod
    def getTypeValueExpression(cls, operand):
        if False:
            while True:
                i = 10
        return '&PyLong_Type'

    def hasSlot(self, slot):
        if False:
            i = 10
            return i + 15
        if slot.startswith('nb_inplace_'):
            return False
        elif slot.startswith('nb_'):
            return slot != 'nb_matrix_multiply'
        elif slot.startswith('sq_'):
            return False
        elif slot == 'tp_richcompare':
            assert False
            return False
        elif slot == 'tp_compare':
            return True
        else:
            assert False, slot

    def getSlotValueExpression(self, operand, slot):
        if False:
            return 10
        if slot == 'tp_richcompare':
            return '(PYTHON_VERSION < 0x300 ? NULL : PyLong_Type.tp_richcompare)'
        return ConcreteTypeBase.getSlotValueExpression(self, operand=operand, slot=slot)

    @classmethod
    def getNewStyleNumberTypeCheckExpression(cls, operand):
        if False:
            return 10
        return '1'

    @staticmethod
    def needsIndexConversion():
        if False:
            while True:
                i = 10
        return False

    def hasPreferredSlot(self, right, slot):
        if False:
            i = 10
            return i + 15
        if right is int_desc:
            return True
        return False

    @staticmethod
    def getLongValueSizeExpression(operand):
        if False:
            for i in range(10):
                print('nop')
        return 'Py_SIZE(%s_long_object)' % operand

    @staticmethod
    def getLongValueIsNegativeTestExpression(operand):
        if False:
            i = 10
            return i + 15
        return 'Py_SIZE(%s_long_object) < 0' % operand

    @staticmethod
    def getLongValueDigitCountExpression(operand):
        if False:
            while True:
                i = 10
        return 'Py_ABS(Py_SIZE(%s_long_object))' % operand

    @staticmethod
    def getLongValueDigitExpression(operand, index):
        if False:
            return 10
        return '%s_long_object->ob_digit[%s]' % (operand, index)

    @staticmethod
    def getLongValueDigitsPointerExpression(operand):
        if False:
            i = 10
            return i + 15
        return '%s_long_object->ob_digit' % operand

    @staticmethod
    def getLongValueMediumValueExpression(operand):
        if False:
            for i in range(10):
                print('nop')
        return 'MEDIUM_VALUE(%s_long_object)' % operand
long_desc = LongDesc()

class ObjectDesc(TypeDescBase):
    type_name = 'object'
    type_desc = 'any Python object'
    type_decl = 'PyObject *'

    def hasSlot(self, slot):
        if False:
            while True:
                i = 10
        assert False

    def getIndexCheckExpression(self, operand):
        if False:
            print('Hello World!')
        return 'Nuitka_Index_Check(%s)' % operand

    def getNewStyleNumberTypeCheckExpression(self, operand):
        if False:
            while True:
                i = 10
        return 'NEW_STYLE_NUMBER_TYPE(%s)' % operand

    def getSlotValueExpression(self, operand, slot):
        if False:
            while True:
                i = 10
        return self._getSlotValueExpression(operand, slot)

    def getSlotValueCheckExpression(self, operand, slot):
        if False:
            i = 10
            return i + 15
        return '(%s) != NULL' % self._getSlotValueExpression(operand, slot)

    @staticmethod
    def getToValueFromBoolExpression(operand):
        if False:
            while True:
                i = 10
        return 'BOOL_FROM(%s)' % operand

    @staticmethod
    def getToValueFromObjectExpression(operand):
        if False:
            print('Hello World!')
        return operand

    @staticmethod
    def getExceptionResultIndicatorValue():
        if False:
            print('Hello World!')
        return 'NULL'

    def getNoSequenceSlotAccessTestCode(self, type_name):
        if False:
            while True:
                i = 10
        return '%s->tp_as_sequence == NULL' % type_name
object_desc = ObjectDesc()

class ConcreteCTypeBase(TypeDescBase):
    """Base class for non-Python (C) concrete types."""

    def hasSlot(self, slot):
        if False:
            return 10
        return False

    def getNoSequenceSlotAccessTestCode(self, type_name):
        if False:
            i = 10
            return i + 15
        assert False, self

    def getNewStyleNumberTypeCheckExpression(self, operand):
        if False:
            return 10
        assert False, self

    @classmethod
    def hasReferenceCounting(cls):
        if False:
            while True:
                i = 10
        return False

    @staticmethod
    def hasPreferredSlot(right, slot):
        if False:
            i = 10
            return i + 15
        return False

class CLongDesc(ConcreteCTypeBase):
    type_name = 'clong'
    type_desc = 'C platform long value'
    type_decl = 'long'

    @classmethod
    def getCheckValueCode(cls, operand):
        if False:
            while True:
                i = 10
        return ''

    @classmethod
    def getTypeValueExpression(cls, operand):
        if False:
            for i in range(10):
                print('nop')
        return 'NULL'

    @classmethod
    def getNewStyleNumberTypeCheckExpression(cls, operand):
        if False:
            for i in range(10):
                print('nop')
        return '0'

    @staticmethod
    def getAsLongValueExpression(operand):
        if False:
            while True:
                i = 10
        return operand

    @staticmethod
    def getAsObjectValueExpression(operand):
        if False:
            for i in range(10):
                print('nop')
        return 'PyLong_FromLong(%s)' % operand

    @staticmethod
    def releaseAsObjectValueStatement(operand):
        if False:
            print('Hello World!')
        return 'Py_DECREF(%s);' % operand

    @staticmethod
    def getLongValueSizeExpression(operand):
        if False:
            while True:
                i = 10
        return '%s_size' % operand

    @staticmethod
    def getLongValueIsNegativeTestExpression(operand):
        if False:
            for i in range(10):
                print('nop')
        return '%s_is_negative' % operand

    @staticmethod
    def getLongValueDigitCountExpression(operand):
        if False:
            for i in range(10):
                print('nop')
        return '%s_digit_count' % operand

    @staticmethod
    def getLongValueDigitExpression(operand, index):
        if False:
            print('Hello World!')
        return '%s_digits[%s]' % (operand, index)

    @staticmethod
    def getLongValueDigitsPointerExpression(operand):
        if False:
            while True:
                i = 10
        return '%s_digits' % operand

    @staticmethod
    def getLongValueMediumValueExpression(operand):
        if False:
            for i in range(10):
                print('nop')
        return '(sdigit)%s' % operand
c_long_desc = CLongDesc()

class CDigitDesc(CLongDesc):
    type_name = 'digit'
    type_desc = 'C platform digit value for long Python objects'
    type_decl = 'long'

    @classmethod
    def getCheckValueCode(cls, operand):
        if False:
            for i in range(10):
                print('nop')
        return 'assert(Py_ABS(%s) < (1 << PyLong_SHIFT));' % operand

    @staticmethod
    def getAsLongValueExpression(operand):
        if False:
            for i in range(10):
                print('nop')
        return '(long)(%s)' % operand

    @staticmethod
    def getLongValueDigitCountExpression(operand):
        if False:
            return 10
        return '(%s == 0 ? 0 : 1)' % operand

    @staticmethod
    def getLongValueSizeExpression(operand):
        if False:
            while True:
                i = 10
        return '(Py_ssize_t)((%(operand)s == 0) ? 0 : ((%(operand)s < 0 ) ? -1 : 1))' % {'operand': operand}

    @staticmethod
    def getLongValueIsNegativeTestExpression(operand):
        if False:
            while True:
                i = 10
        return '%s < 0' % operand

    @staticmethod
    def getLongValueDigitExpression(operand, index):
        if False:
            return 10
        return '(digit)Py_ABS(%s)' % operand

    @staticmethod
    def getLongValueDigitsPointerExpression(operand):
        if False:
            while True:
                i = 10
        return '(digit *)&%s' % operand
c_digit_desc = CDigitDesc()

class CBoolDesc(ConcreteCTypeBase):
    type_name = 'cbool'
    type_desc = 'C platform bool value'
    type_decl = 'bool'

    @classmethod
    def getCheckValueCode(cls, operand):
        if False:
            return 10
        return ''

    @classmethod
    def getTypeValueExpression(cls, operand):
        if False:
            for i in range(10):
                print('nop')
        return 'NULL'

    @classmethod
    def getNewStyleNumberTypeCheckExpression(cls, operand):
        if False:
            while True:
                i = 10
        return '0'

    @staticmethod
    def getAsLongValueExpression(operand):
        if False:
            return 10
        return operand

    @staticmethod
    def getAsObjectValueExpression(operand):
        if False:
            for i in range(10):
                print('nop')
        return 'BOOL_FROM(%s)' % operand

    @staticmethod
    def getToValueFromBoolExpression(operand):
        if False:
            while True:
                i = 10
        return operand

    @staticmethod
    def getToValueFromObjectExpression(operand):
        if False:
            return 10
        return 'CHECK_IF_TRUE(%s) == 1' % operand

    @staticmethod
    def getTakeReferenceStatement(operand):
        if False:
            for i in range(10):
                print('nop')
        return ''

    @staticmethod
    def getExceptionResultIndicatorValue():
        if False:
            i = 10
            return i + 15
        return 'false'
c_bool_desc = CBoolDesc()

class NBoolDesc(ConcreteCTypeBase):
    type_name = 'nbool'
    type_desc = 'Nuitka C bool value'
    type_decl = 'nuitka_bool'

    @classmethod
    def getCheckValueCode(cls, operand):
        if False:
            for i in range(10):
                print('nop')
        return ''

    @classmethod
    def getTypeValueExpression(cls, operand):
        if False:
            for i in range(10):
                print('nop')
        return 'NULL'

    @classmethod
    def getNewStyleNumberTypeCheckExpression(cls, operand):
        if False:
            for i in range(10):
                print('nop')
        return '0'

    @staticmethod
    def getAsLongValueExpression(operand):
        if False:
            return 10
        return operand

    @staticmethod
    def getAsObjectValueExpression(operand):
        if False:
            for i in range(10):
                print('nop')
        return 'BOOL_FROM(%s)' % operand

    @staticmethod
    def getToValueFromBoolExpression(operand):
        if False:
            while True:
                i = 10
        return '%s ? NUITKA_BOOL_TRUE : NUITKA_BOOL_FALSE' % operand

    @classmethod
    def getToValueFromObjectExpression(cls, operand):
        if False:
            for i in range(10):
                print('nop')
        return cls.getToValueFromBoolExpression('CHECK_IF_TRUE(%s)' % operand)

    @staticmethod
    def getTakeReferenceStatement(operand):
        if False:
            return 10
        return ''

    @staticmethod
    def getExceptionResultIndicatorValue():
        if False:
            i = 10
            return i + 15
        return 'NUITKA_BOOL_EXCEPTION'
n_bool_desc = NBoolDesc()

class NVoidDesc(ConcreteCTypeBase):
    type_name = 'nvoid'
    type_desc = 'Nuitka C void value'
    type_decl = 'nuitka_void'

    @classmethod
    def getCheckValueCode(cls, operand):
        if False:
            for i in range(10):
                print('nop')
        return ''

    @classmethod
    def getTypeValueExpression(cls, operand):
        if False:
            print('Hello World!')
        return 'NULL'

    @classmethod
    def getNewStyleNumberTypeCheckExpression(cls, operand):
        if False:
            while True:
                i = 10
        return '0'

    @staticmethod
    def getAsLongValueExpression(operand):
        if False:
            return 10
        assert False
        return operand

    @staticmethod
    def getAsObjectValueExpression(operand):
        if False:
            for i in range(10):
                print('nop')
        assert False
        return 'BOOL_FROM(%s)' % operand

    @staticmethod
    def getToValueFromBoolExpression(operand):
        if False:
            print('Hello World!')
        return 'NUITKA_VOID_OK'

    @classmethod
    def getToValueFromObjectExpression(cls, operand):
        if False:
            while True:
                i = 10
        return 'NUITKA_VOID_OK'

    @staticmethod
    def getTakeReferenceStatement(operand):
        if False:
            while True:
                i = 10
        return ''

    @staticmethod
    def getExceptionResultIndicatorValue():
        if False:
            print('Hello World!')
        return 'NUITKA_VOID_EXCEPTION'
nvoid_desc = NVoidDesc()

class CFloatDesc(ConcreteCTypeBase):
    type_name = 'cfloat'
    type_desc = 'C platform float value'
    type_decl = 'double'

    @classmethod
    def getCheckValueCode(cls, operand):
        if False:
            i = 10
            return i + 15
        return ''

    @classmethod
    def getTypeValueExpression(cls, operand):
        if False:
            i = 10
            return i + 15
        return 'NULL'

    @classmethod
    def getNewStyleNumberTypeCheckExpression(cls, operand):
        if False:
            while True:
                i = 10
        return '0'

    @staticmethod
    def getAsLongValueExpression(operand):
        if False:
            for i in range(10):
                print('nop')
        return operand

    @staticmethod
    def getAsDoubleValueExpression(operand):
        if False:
            print('Hello World!')
        return operand

    @staticmethod
    def getAsObjectValueExpression(operand):
        if False:
            for i in range(10):
                print('nop')
        return 'MAKE_FLOAT_FROM_DOUBLE(%s)' % operand

    @staticmethod
    def releaseAsObjectValueStatement(operand):
        if False:
            for i in range(10):
                print('nop')
        return 'Py_DECREF(%s);' % operand
c_float_desc = CFloatDesc()
related_types = {}

def _addRelatedTypes(type_desc_1, type_desc_2):
    if False:
        return 10
    related_types[type_desc_1] = (type_desc_2,)
    related_types[type_desc_2] = (type_desc_1,)
_addRelatedTypes(int_desc, c_long_desc)
_addRelatedTypes(long_desc, c_digit_desc)
_addRelatedTypes(float_desc, c_float_desc)