""" Code generation for dictionaries.

"""
from nuitka import Options
from nuitka.PythonVersions import python_version
from nuitka.utils.Jinja2 import renderTemplateFromString
from .CallCodes import getCallCodePosArgsQuick
from .CodeHelpers import assignConstantNoneResult, decideConversionCheckNeeded, generateChildExpressionCode, generateChildExpressionsCode, generateExpressionCode, withCleanupFinally, withObjectCodeTemporaryAssignment
from .ErrorCodes import getErrorExitBoolCode, getErrorExitCode, getReleaseCode
from .PythonAPICodes import generateCAPIObjectCode, generateCAPIObjectCode0, makeArgDescFromExpression

def generateBuiltinDictCode(to_name, expression, emit, context):
    if False:
        return 10
    if expression.subnode_pos_arg:
        seq_name = context.allocateTempName('dict_seq')
        generateExpressionCode(to_name=seq_name, expression=expression.subnode_pos_arg, emit=emit, context=context, allow_none=True)
    else:
        seq_name = None
    with withObjectCodeTemporaryAssignment(to_name, 'dict_value', expression, emit, context) as value_name:
        if expression.subnode_pairs:
            if seq_name is None:
                getDictionaryCreationCode(to_name=value_name, pairs=expression.subnode_pairs, emit=emit, context=context)
                dict_name = None
            else:
                dict_name = context.allocateTempName('dict_arg')
                getDictionaryCreationCode(to_name=dict_name, pairs=expression.subnode_pairs, emit=emit, context=context)
        else:
            dict_name = None
        if seq_name is not None:
            emit('%s = TO_DICT(tstate, %s, %s);' % (value_name, seq_name, 'NULL' if dict_name is None else dict_name))
            getErrorExitCode(check_name=value_name, release_names=(seq_name, dict_name), emit=emit, context=context)
            context.addCleanupTempName(value_name)

def generateDictionaryCreationCode(to_name, expression, emit, context):
    if False:
        i = 10
        return i + 15
    with withObjectCodeTemporaryAssignment(to_name, 'dict_result', expression, emit, context) as value_name:
        getDictionaryCreationCode(to_name=value_name, pairs=expression.subnode_pairs, emit=emit, context=context)

def getDictionaryCreationCode(to_name, pairs, emit, context):
    if False:
        return 10
    pairs_count = len(pairs)
    assert pairs_count > 0
    dict_key_name = context.allocateTempName('dict_key')
    dict_value_name = context.allocateTempName('dict_value')
    is_hashable_key = [pair.isKeyKnownToBeHashable() for pair in pairs]
    if all(is_hashable_key):
        for pair in pairs[1:]:
            if pair.mayRaiseException(BaseException):
                needs_exception_exit = True
                break
        else:
            needs_exception_exit = False
    else:
        needs_exception_exit = True

    def generateValueCode(dict_value_name, pair):
        if False:
            print('Hello World!')
        generateExpressionCode(to_name=dict_value_name, expression=pair.getValueNode(), emit=emit, context=context)

    def generateKeyCode(dict_key_name, pair):
        if False:
            while True:
                i = 10
        generateExpressionCode(to_name=dict_key_name, expression=pair.getKeyNode(), emit=emit, context=context)

    def generatePairCode(pair):
        if False:
            while True:
                i = 10
        if python_version < 848:
            generateValueCode(dict_value_name, pair)
            generateKeyCode(dict_key_name, pair)
        else:
            generateKeyCode(dict_key_name, pair)
            generateValueCode(dict_value_name, pair)
        key_needs_release = context.needsCleanup(dict_key_name)
        if key_needs_release:
            context.removeCleanupTempName(dict_key_name)
        value_needs_release = context.needsCleanup(dict_value_name)
        if value_needs_release:
            context.removeCleanupTempName(dict_value_name)
        return (key_needs_release, value_needs_release)
    (key_needs_release, value_needs_release) = generatePairCode(pairs[0])
    emit('%s = _PyDict_NewPresized( %d );' % (to_name, pairs_count))
    with withCleanupFinally('dict_build', to_name, needs_exception_exit, emit, context) as guarded_emit:
        emit = guarded_emit.emit
        for (count, pair) in enumerate(pairs):
            if count > 0:
                (key_needs_release, value_needs_release) = generatePairCode(pair)
            needs_check = not is_hashable_key[count]
            res_name = context.getIntResName()
            emit('%s = PyDict_SetItem(%s, %s, %s);' % (res_name, to_name, dict_key_name, dict_value_name))
            if value_needs_release:
                emit('Py_DECREF(%s);' % dict_value_name)
            if key_needs_release:
                emit('Py_DECREF(%s);' % dict_key_name)
            getErrorExitBoolCode(condition='%s != 0' % res_name, needs_check=needs_check, emit=emit, context=context)

def generateDictOperationUpdateCode(statement, emit, context):
    if False:
        i = 10
        return i + 15
    value_arg_name = context.allocateTempName('dictupdate_value', unique=True)
    generateExpressionCode(to_name=value_arg_name, expression=statement.subnode_value, emit=emit, context=context)
    dict_arg_name = context.allocateTempName('dictupdate_dict', unique=True)
    generateExpressionCode(to_name=dict_arg_name, expression=statement.subnode_dict_arg, emit=emit, context=context)
    with context.withCurrentSourceCodeReference(statement.getSourceReference()):
        res_name = context.getIntResName()
        emit('assert(PyDict_Check(%s));' % dict_arg_name)
        emit('%s = PyDict_Update(%s, %s);' % (res_name, dict_arg_name, value_arg_name))
        getErrorExitBoolCode(condition='%s != 0' % res_name, release_names=(dict_arg_name, value_arg_name), needs_check=statement.mayRaiseException(BaseException), emit=emit, context=context)

def generateDictOperationItemCode(to_name, expression, emit, context):
    if False:
        return 10
    (dict_name, key_name) = generateChildExpressionsCode(expression=expression, emit=emit, context=context)
    with withObjectCodeTemporaryAssignment(to_name, 'dict_value', expression, emit, context) as value_name:
        emit('%s = DICT_GET_ITEM_WITH_ERROR(tstate, %s, %s);' % (value_name, dict_name, key_name))
        getErrorExitCode(check_name=value_name, release_names=(dict_name, key_name), needs_check=expression.mayRaiseException(BaseException), emit=emit, context=context)
        context.addCleanupTempName(value_name)

def generateDictOperationGet2Code(to_name, expression, emit, context):
    if False:
        while True:
            i = 10
    (dict_name, key_name) = generateChildExpressionsCode(expression=expression, emit=emit, context=context)
    with withObjectCodeTemporaryAssignment(to_name, 'dict_value', expression, emit, context) as value_name:
        emit(renderTemplateFromString('\n{% if expression.known_hashable_key %}\n%(value_name)s = DICT_GET_ITEM0(tstate, %(dict_name)s, %(key_name)s);\nif (%(value_name)s == NULL) {\n{% else %}\n%(value_name)s = DICT_GET_ITEM_WITH_HASH_ERROR0(tstate, %(dict_name)s, %(key_name)s);\nif (%(value_name)s == NULL && !HAS_ERROR_OCCURRED(tstate)) {\n{% endif %}\n    %(value_name)s = Py_None;\n}\n', expression=expression) % {'value_name': value_name, 'dict_name': dict_name, 'key_name': key_name})
        getErrorExitCode(check_name=value_name, release_names=(dict_name, key_name), needs_check=not expression.known_hashable_key, emit=emit, context=context)

def generateDictOperationGet3Code(to_name, expression, emit, context):
    if False:
        while True:
            i = 10
    (dict_name, key_name, default_name) = generateChildExpressionsCode(expression=expression, emit=emit, context=context)
    with withObjectCodeTemporaryAssignment(to_name, 'dict_value', expression, emit, context) as value_name:
        emit(renderTemplateFromString('\n{% if expression.known_hashable_key %}\n%(value_name)s = DICT_GET_ITEM1(tstate, %(dict_name)s, %(key_name)s);\nif (%(value_name)s == NULL) {\n{% else %}\n%(value_name)s = DICT_GET_ITEM_WITH_HASH_ERROR1(tstate, %(dict_name)s, %(key_name)s);\nif (%(value_name)s == NULL && !HAS_ERROR_OCCURRED(tstate)) {\n{% endif %}\n    %(value_name)s = %(default_name)s;\n    Py_INCREF(%(value_name)s);\n}\n\n', expression=expression) % {'value_name': value_name, 'dict_name': dict_name, 'key_name': key_name, 'default_name': default_name})
        getErrorExitCode(check_name=value_name, release_names=(dict_name, key_name, default_name), needs_check=not expression.known_hashable_key, emit=emit, context=context)
        context.addCleanupTempName(value_name)

def generateDictOperationSetdefault2Code(to_name, expression, emit, context):
    if False:
        i = 10
        return i + 15
    generateCAPIObjectCode(to_name=to_name, capi='DICT_SETDEFAULT2', tstate=True, arg_desc=makeArgDescFromExpression(expression), may_raise=not expression.known_hashable_key, conversion_check=decideConversionCheckNeeded(to_name, expression), source_ref=expression.getCompatibleSourceReference(), emit=emit, context=context)

def generateDictOperationSetdefault3Code(to_name, expression, emit, context):
    if False:
        print('Hello World!')
    generateCAPIObjectCode(to_name=to_name, capi='DICT_SETDEFAULT3', tstate=True, arg_desc=makeArgDescFromExpression(expression), may_raise=not expression.known_hashable_key, conversion_check=decideConversionCheckNeeded(to_name, expression), source_ref=expression.getCompatibleSourceReference(), emit=emit, context=context)

def generateDictOperationPop2Code(to_name, expression, emit, context):
    if False:
        i = 10
        return i + 15
    generateCAPIObjectCode(to_name=to_name, capi='DICT_POP2', tstate=True, arg_desc=makeArgDescFromExpression(expression), may_raise=expression.mayRaiseException(BaseException), conversion_check=decideConversionCheckNeeded(to_name, expression), source_ref=expression.getCompatibleSourceReference(), emit=emit, context=context)

def generateDictOperationPop3Code(to_name, expression, emit, context):
    if False:
        for i in range(10):
            print('nop')
    generateCAPIObjectCode(to_name=to_name, capi='DICT_POP3', tstate=True, arg_desc=makeArgDescFromExpression(expression), may_raise=not expression.known_hashable_key, conversion_check=decideConversionCheckNeeded(to_name, expression), source_ref=expression.getCompatibleSourceReference(), emit=emit, context=context)

def generateDictOperationPopitemCode(to_name, expression, emit, context):
    if False:
        i = 10
        return i + 15
    generateCAPIObjectCode(to_name=to_name, capi='DICT_POPITEM', tstate=True, arg_desc=makeArgDescFromExpression(expression), may_raise=expression.mayRaiseException(BaseException), conversion_check=decideConversionCheckNeeded(to_name, expression), source_ref=expression.getCompatibleSourceReference(), emit=emit, context=context)

def _generateDictOperationUpdateCommonCode(dict_name, iterable_name, release_dict, expression, emit, context):
    if False:
        while True:
            i = 10
    res_name = context.getIntResName()
    emit('assert(PyDict_Check(%s));' % dict_name)
    emit(renderTemplateFromString('\n{% if has_keys_attribute == None %}\nif (HAS_ATTR_BOOL(tstate, %(iterable_name)s, const_str_plain_keys)){\n    %(res_name)s = PyDict_Merge(%(dict_name)s, %(iterable_name)s, 1);\n} else {\n    %(res_name)s = PyDict_MergeFromSeq2(%(dict_name)s, %(iterable_name)s, 1);\n}\n{% elif has_keys_attribute == True %}\n    %(res_name)s = PyDict_Merge(%(dict_name)s, %(iterable_name)s, 1);\n{% else %}\n    %(res_name)s = PyDict_MergeFromSeq2(%(dict_name)s, %(iterable_name)s, 1);\n{% endif %}\n', has_keys_attribute=expression.subnode_iterable.isKnownToHaveAttribute('keys')) % {'res_name': res_name, 'dict_name': dict_name, 'iterable_name': iterable_name})
    if release_dict:
        release_names = (dict_name, iterable_name)
    else:
        release_names = (iterable_name,)
    getErrorExitBoolCode(condition='%s != 0' % res_name, release_names=release_names, needs_check=expression.mayRaiseException(BaseException), emit=emit, context=context)

def generateDictOperationUpdate2Code(to_name, expression, emit, context):
    if False:
        for i in range(10):
            print('nop')
    (dict_name, iterable_name) = generateChildExpressionsCode(expression=expression, emit=emit, context=context)
    _generateDictOperationUpdateCommonCode(dict_name=dict_name, iterable_name=iterable_name, release_dict=True, expression=expression, emit=emit, context=context)
    assignConstantNoneResult(to_name, emit, context)

def generateDictOperationUpdate3Code(to_name, expression, emit, context):
    if False:
        while True:
            i = 10
    dict_name = generateChildExpressionCode(expression=expression.subnode_dict_arg, emit=emit, context=context)
    emit('assert(PyDict_Check(%s));' % dict_name)
    if expression.isExpressionDictOperationUpdate3():
        iterable_name = generateChildExpressionCode(expression=expression.subnode_iterable, emit=emit, context=context)
        _generateDictOperationUpdateCommonCode(dict_name=dict_name, iterable_name=iterable_name, release_dict=False, expression=expression, emit=emit, context=context)
    dict_key_name = context.allocateTempName('dictupdate_key')
    dict_value_name = context.allocateTempName('dictupdate_value')
    res_name = context.getIntResName()
    for (count, pair) in enumerate(expression.subnode_pairs):
        generateExpressionCode(to_name=dict_key_name, expression=pair.getKeyNode(), emit=emit, context=context)
        generateExpressionCode(to_name=dict_value_name, expression=pair.getValueNode(), emit=emit, context=context)
        emit('%s = PyDict_SetItem(%s, %s, %s);' % (res_name, dict_name, dict_key_name, dict_value_name))
        getErrorExitBoolCode(condition='%s != 0' % res_name, needs_check=not expression.subnode_pairs[count].isKnownToBeHashable(), release_names=(dict_key_name, dict_value_name), emit=emit, context=context)
    getReleaseCode(release_name=dict_name, emit=emit, context=context)
    assignConstantNoneResult(to_name, emit, context)

def generateDictOperationCopyCode(to_name, expression, emit, context):
    if False:
        while True:
            i = 10
    generateCAPIObjectCode(to_name=to_name, capi='DICT_COPY', tstate=False, arg_desc=(('dict_arg', expression.subnode_dict_arg),), may_raise=False, conversion_check=decideConversionCheckNeeded(to_name, expression), source_ref=expression.getCompatibleSourceReference(), emit=emit, context=context)

def generateDictOperationClearCode(to_name, expression, emit, context):
    if False:
        for i in range(10):
            print('nop')
    generateCAPIObjectCode0(to_name=None, capi='DICT_CLEAR', tstate=False, arg_desc=(('dict_arg', expression.subnode_dict_arg),), may_raise=False, conversion_check=decideConversionCheckNeeded(to_name, expression), source_ref=expression.getCompatibleSourceReference(), emit=emit, context=context)
    assignConstantNoneResult(to_name, emit, context)

def generateDictOperationItemsCode(to_name, expression, emit, context):
    if False:
        for i in range(10):
            print('nop')
    generateCAPIObjectCode(to_name=to_name, capi='DICT_ITEMS', tstate=False, arg_desc=(('dict_arg', expression.subnode_dict_arg),), may_raise=False, conversion_check=decideConversionCheckNeeded(to_name, expression), source_ref=expression.getCompatibleSourceReference(), emit=emit, context=context)

def generateDictOperationIteritemsCode(to_name, expression, emit, context):
    if False:
        return 10
    generateCAPIObjectCode(to_name=to_name, capi='DICT_ITERITEMS', tstate=False, arg_desc=(('dict_arg', expression.subnode_dict_arg),), may_raise=expression.mayRaiseException(BaseException), conversion_check=decideConversionCheckNeeded(to_name, expression), source_ref=expression.getCompatibleSourceReference(), emit=emit, context=context)

def generateDictOperationViewitemsCode(to_name, expression, emit, context):
    if False:
        return 10
    generateCAPIObjectCode(to_name=to_name, capi='DICT_VIEWITEMS', tstate=False, arg_desc=(('dict_arg', expression.subnode_dict_arg),), may_raise=False, conversion_check=decideConversionCheckNeeded(to_name, expression), source_ref=expression.getCompatibleSourceReference(), emit=emit, context=context)

def generateDictOperationKeysCode(to_name, expression, emit, context):
    if False:
        i = 10
        return i + 15
    generateCAPIObjectCode(to_name=to_name, capi='DICT_KEYS', tstate=False, arg_desc=(('dict_arg', expression.subnode_dict_arg),), may_raise=False, conversion_check=decideConversionCheckNeeded(to_name, expression), source_ref=expression.getCompatibleSourceReference(), emit=emit, context=context)

def generateDictOperationIterkeysCode(to_name, expression, emit, context):
    if False:
        while True:
            i = 10
    generateCAPIObjectCode(to_name=to_name, capi='DICT_ITERKEYS', tstate=False, arg_desc=(('dict_arg', expression.subnode_dict_arg),), may_raise=False, conversion_check=decideConversionCheckNeeded(to_name, expression), source_ref=expression.getCompatibleSourceReference(), emit=emit, context=context)

def generateDictOperationViewkeysCode(to_name, expression, emit, context):
    if False:
        while True:
            i = 10
    generateCAPIObjectCode(to_name=to_name, capi='DICT_VIEWKEYS', tstate=False, arg_desc=(('dict_arg', expression.subnode_dict_arg),), may_raise=False, conversion_check=decideConversionCheckNeeded(to_name, expression), source_ref=expression.getCompatibleSourceReference(), emit=emit, context=context)

def generateDictOperationValuesCode(to_name, expression, emit, context):
    if False:
        return 10
    generateCAPIObjectCode(to_name=to_name, capi='DICT_VALUES', tstate=False, arg_desc=(('dict_arg', expression.subnode_dict_arg),), may_raise=False, conversion_check=decideConversionCheckNeeded(to_name, expression), source_ref=expression.getCompatibleSourceReference(), emit=emit, context=context)

def generateDictOperationFromkeysRefCode(to_name, expression, emit, context):
    if False:
        i = 10
        return i + 15
    with withObjectCodeTemporaryAssignment(to_name, 'dict_fromkeys_function_ref', expression, emit, context) as result_name:
        emit('%s = dict_builtin_fromkeys;' % result_name)

def generateDictOperationFromkeys2Code(to_name, expression, emit, context):
    if False:
        while True:
            i = 10
    (iterable_name,) = generateChildExpressionsCode(expression=expression, emit=emit, context=context)
    with withObjectCodeTemporaryAssignment(to_name, 'dict_fromkeys_value', expression, emit, context) as result_name:
        getCallCodePosArgsQuick(to_name=result_name, called_name='dict_builtin_fromkeys', expression=expression, arg_names=(iterable_name,), emit=emit, context=context)

def generateDictOperationFromkeys3Code(to_name, expression, emit, context):
    if False:
        print('Hello World!')
    (iterable_name, value_name) = generateChildExpressionsCode(expression=expression, emit=emit, context=context)
    with withObjectCodeTemporaryAssignment(to_name, 'dict_fromkeys_value', expression, emit, context) as result_name:
        getCallCodePosArgsQuick(to_name=result_name, called_name='dict_builtin_fromkeys', expression=expression, arg_names=(iterable_name, value_name), emit=emit, context=context)

def generateDictOperationItervaluesCode(to_name, expression, emit, context):
    if False:
        print('Hello World!')
    generateCAPIObjectCode(to_name=to_name, capi='DICT_ITERVALUES', tstate=False, arg_desc=(('dict_arg', expression.subnode_dict_arg),), may_raise=False, conversion_check=decideConversionCheckNeeded(to_name, expression), source_ref=expression.getCompatibleSourceReference(), emit=emit, context=context)

def generateDictOperationViewvaluesCode(to_name, expression, emit, context):
    if False:
        while True:
            i = 10
    generateCAPIObjectCode(to_name=to_name, capi='DICT_VIEWVALUES', tstate=False, arg_desc=(('dict_arg', expression.subnode_dict_arg),), may_raise=False, conversion_check=decideConversionCheckNeeded(to_name, expression), source_ref=expression.getCompatibleSourceReference(), emit=emit, context=context)

def generateDictOperationInCode(to_name, expression, emit, context):
    if False:
        print('Hello World!')
    inverted = expression.isExpressionDictOperationNotIn()
    (dict_name, key_name) = generateChildExpressionsCode(expression=expression, emit=emit, context=context)
    if expression.isExpressionDictOperationHaskey():
        (dict_name, key_name) = (key_name, dict_name)
    res_name = context.getIntResName()
    emit('%s = DICT_HAS_ITEM(tstate, %s, %s);' % (res_name, key_name, dict_name))
    getErrorExitBoolCode(condition='%s == -1' % res_name, release_names=(dict_name, key_name), needs_check=expression.known_hashable_key is not True, emit=emit, context=context)
    to_name.getCType().emitAssignmentCodeFromBoolCondition(to_name=to_name, condition='%s %s 0' % (res_name, '==' if inverted else '!='), emit=emit)

def generateDictOperationSetCode(statement, emit, context):
    if False:
        i = 10
        return i + 15
    value_arg_name = context.allocateTempName('dictset_value', unique=True)
    generateExpressionCode(to_name=value_arg_name, expression=statement.subnode_value, emit=emit, context=context)
    dict_arg_name = context.allocateTempName('dictset_dict', unique=True)
    generateExpressionCode(to_name=dict_arg_name, expression=statement.subnode_dict_arg, emit=emit, context=context)
    key_arg_name = context.allocateTempName('dictset_key', unique=True)
    generateExpressionCode(to_name=key_arg_name, expression=statement.subnode_key, emit=emit, context=context)
    context.setCurrentSourceCodeReference(statement.getSourceReference())
    res_name = context.getIntResName()
    emit('assert(PyDict_CheckExact(%s));\n%s = PyDict_SetItem(%s, %s, %s);\n' % (dict_arg_name, res_name, dict_arg_name, key_arg_name, value_arg_name))
    getErrorExitBoolCode(condition='%s != 0' % res_name, release_names=(value_arg_name, dict_arg_name, key_arg_name), emit=emit, needs_check=not statement.subnode_key.isKnownToBeHashable(), context=context)

def generateDictOperationSetCodeKeyValue(statement, emit, context):
    if False:
        i = 10
        return i + 15
    key_arg_name = context.allocateTempName('dictset38_key')
    generateExpressionCode(to_name=key_arg_name, expression=statement.subnode_key, emit=emit, context=context)
    value_arg_name = context.allocateTempName('dictset38_value')
    generateExpressionCode(to_name=value_arg_name, expression=statement.subnode_value, emit=emit, context=context)
    dict_arg_name = context.allocateTempName('dictset38_dict')
    generateExpressionCode(to_name=dict_arg_name, expression=statement.subnode_dict_arg, emit=emit, context=context)
    context.setCurrentSourceCodeReference(statement.getSourceReference())
    res_name = context.getIntResName()
    emit('assert(PyDict_CheckExact(%s));\n%s = PyDict_SetItem(%s, %s, %s);\n' % (dict_arg_name, res_name, dict_arg_name, key_arg_name, value_arg_name))
    getErrorExitBoolCode(condition='%s != 0' % res_name, release_names=(value_arg_name, dict_arg_name, key_arg_name), emit=emit, needs_check=not statement.subnode_key.isKnownToBeHashable(), context=context)

def generateDictOperationRemoveCode(statement, emit, context):
    if False:
        i = 10
        return i + 15
    dict_arg_name = context.allocateTempName('dictdel_dict', unique=True)
    generateExpressionCode(to_name=dict_arg_name, expression=statement.subnode_dict_arg, emit=emit, context=context)
    key_arg_name = context.allocateTempName('dictdel_key', unique=True)
    generateExpressionCode(to_name=key_arg_name, expression=statement.subnode_key, emit=emit, context=context)
    with context.withCurrentSourceCodeReference(statement.subnode_key.getSourceReference() if Options.is_full_compat else statement.getSourceReference()):
        res_name = context.getBoolResName()
        emit('%s = DICT_REMOVE_ITEM(%s, %s);' % (res_name, dict_arg_name, key_arg_name))
        getErrorExitBoolCode(condition='%s == false' % res_name, release_names=(dict_arg_name, key_arg_name), needs_check=statement.mayRaiseException(BaseException), emit=emit, context=context)