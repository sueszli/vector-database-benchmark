""" Code generation for calls.

The different kinds of calls get dedicated code. Most notable, calls with
only positional arguments, are attempted through helpers that might be
able to execute them without creating the argument dictionary at all.

"""
from nuitka.Constants import isMutable
from nuitka.utils.Jinja2 import getTemplateC
from .CodeHelpers import generateChildExpressionCode, generateExpressionCode, withObjectCodeTemporaryAssignment
from .ErrorCodes import getErrorExitCode
from .LineNumberCodes import emitLineNumberUpdateCode
from .templates.CodeTemplatesModules import template_header_guard, template_helper_impl_decl

def _generateCallCodePosOnly(to_name, expression, called_name, called_attribute_name, emit, context):
    if False:
        return 10
    assert called_name is not None
    call_args = expression.subnode_args
    if call_args is None or call_args.isExpressionConstantRef():
        context.setCurrentSourceCodeReference(expression.getCompatibleSourceReference())
        if call_args is not None:
            call_args_value = call_args.getCompileTimeConstant()
        else:
            call_args_value = ()
        assert type(call_args_value) is tuple
        if call_args is not None and call_args.isMutable():
            call_arg_names = []
            for call_arg_element in call_args_value:
                call_arg_name = context.allocateTempName('call_arg_element')
                call_arg_name.getCType().emitAssignmentCodeFromConstant(to_name=call_arg_name, constant=call_arg_element, may_escape=True, emit=emit, context=context)
                call_arg_names.append(call_arg_name)
            if called_attribute_name is None:
                getCallCodePosArgsQuick(to_name=to_name, called_name=called_name, arg_names=call_arg_names, expression=expression, emit=emit, context=context)
            else:
                _getInstanceCallCodePosArgsQuick(to_name=to_name, called_name=called_name, called_attribute_name=called_attribute_name, expression=expression, arg_names=call_arg_names, emit=emit, context=context)
        elif call_args_value:
            if called_attribute_name is None:
                _getCallCodeFromTuple(to_name=to_name, called_name=called_name, expression=expression, args_value=call_args_value, emit=emit, context=context)
            else:
                _getInstanceCallCodeFromTuple(to_name=to_name, called_name=called_name, called_attribute_name=called_attribute_name, expression=expression, arg_tuple=context.getConstantCode(constant=call_args_value), arg_size=len(call_args_value), emit=emit, context=context)
        elif called_attribute_name is None:
            getCallCodeNoArgs(to_name=to_name, called_name=called_name, expression=expression, emit=emit, context=context)
        else:
            _getInstanceCallCodeNoArgs(to_name=to_name, called_name=called_name, called_attribute_name=called_attribute_name, expression=expression, emit=emit, context=context)
    elif call_args.isExpressionMakeTuple():
        call_arg_names = []
        for call_arg_element in call_args.subnode_elements:
            call_arg_name = generateChildExpressionCode(child_name=call_args.getChildName() + '_element', expression=call_arg_element, emit=emit, context=context)
            call_arg_names.append(call_arg_name)
        if called_attribute_name is None:
            getCallCodePosArgsQuick(to_name=to_name, called_name=called_name, expression=expression, arg_names=call_arg_names, emit=emit, context=context)
        else:
            _getInstanceCallCodePosArgsQuick(to_name=to_name, called_name=called_name, called_attribute_name=called_attribute_name, expression=expression, arg_names=call_arg_names, emit=emit, context=context)
    else:
        args_name = generateChildExpressionCode(expression=call_args, emit=emit, context=context)
        context.setCurrentSourceCodeReference(expression.getCompatibleSourceReference())
        if called_attribute_name is None:
            _getCallCodePosArgs(to_name=to_name, called_name=called_name, expression=expression, args_name=args_name, emit=emit, context=context)
        else:
            _getInstanceCallCodePosArgs(to_name=to_name, called_name=called_name, called_attribute_name=called_attribute_name, expression=expression, args_name=args_name, emit=emit, context=context)

def _getCallCodeKwSplitFromConstant(to_name, expression, call_kw, called_name, called_attribute_name, emit, context):
    if False:
        i = 10
        return i + 15
    assert called_name is not None
    assert called_attribute_name is None
    kw_items = tuple(call_kw.getCompileTimeConstant().items())
    values = tuple((item[1] for item in kw_items))
    kw_names = tuple((item[0] for item in kw_items))
    if isMutable(values):
        args_kwsplit_name = context.allocateTempName('call_args_kwsplit')
        args_kwsplit_name.getCType().emitAssignmentCodeFromConstant(to_name=args_kwsplit_name, constant=values, may_escape=True, emit=emit, context=context)
        split_name = args_kwsplit_name
    else:
        args_kwsplit_name = context.getConstantCode(values)
        split_name = None
    emitLineNumberUpdateCode(expression, emit, context)
    emit('%s = CALL_FUNCTION_WITH_NO_ARGS_KWSPLIT(tstate, %s, &PyTuple_GET_ITEM(%s, 0), %s);' % (to_name, called_name, args_kwsplit_name, context.getConstantCode(kw_names)))
    getErrorExitCode(check_name=to_name, release_names=(called_name, split_name), needs_check=expression.mayRaiseExceptionOperation(), emit=emit, context=context)
    context.addCleanupTempName(to_name)

def getCallCodeKwSplit(to_name, called_name, kw_names, dict_value_names, needs_check, emit, context):
    if False:
        i = 10
        return i + 15
    emit('{\n    PyObject *kw_values[%(kw_size)d] = {%(kw_value_names)s};\n\n    %(to_name)s = CALL_FUNCTION_WITH_NO_ARGS_KWSPLIT(tstate, %(called_name)s, kw_values, %(kw_names)s);\n}\n' % {'to_name': to_name, 'kw_value_names': ', '.join((str(dict_value_name) for dict_value_name in dict_value_names)), 'kw_size': len(kw_names), 'called_name': called_name, 'kw_names': context.getConstantCode(tuple(kw_names))})
    getErrorExitCode(check_name=to_name, release_names=(called_name,) + tuple(dict_value_names), needs_check=needs_check, emit=emit, context=context)
    context.addCleanupTempName(to_name)

def getCallCodeKwPairs(to_name, expression, pairs, called_name, called_attribute_name, emit, context):
    if False:
        while True:
            i = 10
    assert called_name is not None
    assert called_attribute_name is None
    context.setCurrentSourceCodeReference(expression.getCompatibleSourceReference())
    kw_names = []
    dict_value_names = []
    for (count, pair) in enumerate(pairs):
        kw_names.append(pair.getKeyCompileTimeConstant())
        dict_value_name = context.allocateTempName('kw_call_value_%d' % count)
        generateExpressionCode(to_name=dict_value_name, expression=pair.getValueNode(), emit=emit, context=context, allow_none=False)
        dict_value_names.append(dict_value_name)
    emitLineNumberUpdateCode(expression, emit, context)
    assert len(kw_names) == len(pairs)
    if kw_names:
        getCallCodeKwSplit(to_name=to_name, called_name=called_name, kw_names=kw_names, dict_value_names=dict_value_names, needs_check=expression.mayRaiseExceptionOperation(), emit=emit, context=context)
    else:
        getCallCodeNoArgs(to_name=to_name, called_name=called_name, expression=expression, emit=emit, context=context)

def _generateCallCodeKwDict(to_name, expression, call_kw, called_name, called_attribute_name, emit, context):
    if False:
        i = 10
        return i + 15
    assert called_name is not None
    assert called_attribute_name is None
    context.setCurrentSourceCodeReference(expression.getCompatibleSourceReference())
    kw_dict_name = context.allocateTempName('kw_dict')
    generateExpressionCode(to_name=kw_dict_name, expression=call_kw, emit=emit, context=context, allow_none=False)
    emitLineNumberUpdateCode(expression, emit, context)
    emit('%(to_name)s = CALL_FUNCTION_WITH_KEYARGS(tstate, %(called_name)s, %(kw_dict_name)s);\n' % {'to_name': to_name, 'kw_dict_name': kw_dict_name, 'called_name': called_name})
    getErrorExitCode(check_name=to_name, release_names=(called_name, kw_dict_name), needs_check=expression.mayRaiseExceptionOperation(), emit=emit, context=context)
    context.addCleanupTempName(to_name)

def generateCallCode(to_name, expression, emit, context):
    if False:
        for i in range(10):
            print('nop')
    called = expression.subnode_called
    call_kw = expression.subnode_kwargs
    call_args = expression.subnode_args
    if called.isExpressionAttributeLookup() and (not called.isExpressionAttributeLookupSpecial()) and (called.getAttributeName() not in ('__class__', '__dict__')) and (call_args is None or not call_args.mayHaveSideEffects() or (not called.mayHaveSideEffects())) and (call_kw is None):
        called_name = context.allocateTempName('called_instance')
        generateExpressionCode(to_name=called_name, expression=called.subnode_expression, emit=emit, context=context)
        called_attribute_name = context.getConstantCode(constant=called.getAttributeName())
    else:
        called_attribute_name = None
        called_name = generateChildExpressionCode(expression=called, emit=emit, context=context)
    with withObjectCodeTemporaryAssignment(to_name, 'call_result', expression, emit, context) as result_name:
        if call_kw is None or call_kw.isExpressionConstantDictEmptyRef():
            _generateCallCodePosOnly(to_name=result_name, called_name=called_name, called_attribute_name=called_attribute_name, expression=expression, emit=emit, context=context)
        else:
            call_args = expression.subnode_args
            if call_args is None or call_args.isExpressionConstantTupleEmptyRef():
                if call_kw.isExpressionConstantDictRef():
                    assert call_kw.isMappingWithConstantStringKeys()
                    _getCallCodeKwSplitFromConstant(to_name=result_name, called_name=called_name, called_attribute_name=called_attribute_name, expression=expression, call_kw=call_kw, emit=emit, context=context)
                elif call_kw.isExpressionMakeDict() and call_kw.isMappingWithConstantStringKeys():
                    getCallCodeKwPairs(to_name=result_name, called_name=called_name, called_attribute_name=called_attribute_name, expression=expression, pairs=call_kw.subnode_pairs, emit=emit, context=context)
                else:
                    _generateCallCodeKwDict(to_name=result_name, called_name=called_name, called_attribute_name=called_attribute_name, expression=expression, call_kw=call_kw, emit=emit, context=context)
            elif call_kw.isExpressionConstantDictRef() and call_args.isExpressionConstantTupleRef():
                _getCallCodePosConstantKeywordConstArgs(to_name=result_name, called_name=called_name, expression=expression, call_args=call_args, call_kw=call_kw, emit=emit, context=context)
            elif call_kw.isExpressionMakeDict() and call_args.isExpressionConstantTupleRef():
                _getCallCodePosConstKeywordVariableArgs(to_name=result_name, called_name=called_name, expression=expression, call_args=call_args, call_kw=call_kw, emit=emit, context=context)
            elif call_kw.isExpressionMakeDict() and call_args.isExpressionMakeTuple():
                getCallCodePosVariableKeywordVariableArgs(to_name=result_name, expression=expression, called_name=called_name, call_args=call_args.subnode_elements, pairs=call_kw.subnode_pairs, emit=emit, context=context)
            else:
                call_args_name = generateChildExpressionCode(expression=call_args, emit=emit, context=context)
                call_kw_name = generateChildExpressionCode(expression=call_kw, emit=emit, context=context)
                context.setCurrentSourceCodeReference(expression.getCompatibleSourceReference())
                _getCallCodePosKeywordArgs(to_name=result_name, called_name=called_name, expression=expression, call_args_name=call_args_name, call_kw_name=call_kw_name, emit=emit, context=context)

def getCallCodeNoArgs(to_name, called_name, expression, emit, context):
    if False:
        while True:
            i = 10
    emitLineNumberUpdateCode(expression, emit, context)
    emit('%s = CALL_FUNCTION_NO_ARGS(tstate, %s);' % (to_name, called_name))
    getErrorExitCode(check_name=to_name, release_name=called_name, needs_check=expression.mayRaiseExceptionOperation(), emit=emit, context=context)
    context.addCleanupTempName(to_name)

def _getInstanceCallCodeNoArgs(to_name, called_name, called_attribute_name, expression, emit, context):
    if False:
        while True:
            i = 10
    emitLineNumberUpdateCode(expression, emit, context)
    emit('%s = CALL_METHOD_NO_ARGS(tstate, %s, %s);' % (to_name, called_name, called_attribute_name))
    getErrorExitCode(check_name=to_name, release_names=(called_name, called_attribute_name), needs_check=expression.mayRaiseExceptionOperation(), emit=emit, context=context)
    context.addCleanupTempName(to_name)
quick_calls_used = set()
quick_tuple_calls_used = set()
quick_instance_calls_used = set()
quick_mixed_calls_used = set()

def _getInstanceCallCodePosArgsQuick(to_name, called_name, called_attribute_name, expression, arg_names, emit, context):
    if False:
        while True:
            i = 10
    arg_size = len(arg_names)
    assert arg_size > 0
    emitLineNumberUpdateCode(expression, emit, context)
    if arg_size == 1:
        emit('%s = CALL_METHOD_WITH_SINGLE_ARG(tstate, %s, %s, %s);' % (to_name, called_name, called_attribute_name, arg_names[0]))
    else:
        quick_instance_calls_used.add(arg_size)
        emit('{\n    PyObject *call_args[] = {%(call_args)s};\n    %(to_name)s = CALL_METHOD_WITH_ARGS%(arg_size)d(\n        tstate,\n        %(called_name)s,\n        %(called_attribute_name)s,\n        call_args\n    );\n}\n' % {'call_args': ', '.join((str(arg_name) for arg_name in arg_names)), 'to_name': to_name, 'arg_size': arg_size, 'called_name': called_name, 'called_attribute_name': called_attribute_name})
    getErrorExitCode(check_name=to_name, release_names=[called_name] + arg_names, needs_check=expression.mayRaiseExceptionOperation(), emit=emit, context=context)
    context.addCleanupTempName(to_name)

def getCallCodePosArgsQuick(to_name, called_name, arg_names, expression, emit, context):
    if False:
        while True:
            i = 10
    arg_size = len(arg_names)
    assert arg_size > 0
    emitLineNumberUpdateCode(expression, emit, context)
    if arg_size == 1:
        emit('%s = CALL_FUNCTION_WITH_SINGLE_ARG(tstate, %s, %s);' % (to_name, called_name, arg_names[0]))
    else:
        quick_calls_used.add(arg_size)
        emit('{\n    PyObject *call_args[] = {%s};\n    %s = CALL_FUNCTION_WITH_ARGS%d(tstate, %s, call_args);\n}\n' % (', '.join((str(arg_name) for arg_name in arg_names)), to_name, arg_size, called_name))
    getErrorExitCode(check_name=to_name, release_names=[called_name] + list(arg_names), needs_check=expression.mayRaiseExceptionOperation(), emit=emit, context=context)
    context.addCleanupTempName(to_name)

def _getInstanceCallCodeFromTuple(to_name, called_name, called_attribute_name, expression, arg_tuple, arg_size, emit, context):
    if False:
        i = 10
        return i + 15
    quick_instance_calls_used.add(arg_size)
    assert arg_size > 0
    emitLineNumberUpdateCode(expression, emit, context)
    if arg_size == 1:
        template = '%(to_name)s = CALL_METHOD_WITH_SINGLE_ARG(\n    tstate,\n    %(called_name)s,\n    %(called_attribute_name)s,\n    PyTuple_GET_ITEM(%(arg_tuple)s, 0)\n);\n'
    else:
        template = '%(to_name)s = CALL_METHOD_WITH_ARGS%(arg_size)d(\n    tstate,\n    %(called_name)s,\n    %(called_attribute_name)s,\n    &PyTuple_GET_ITEM(%(arg_tuple)s, 0)\n);\n'
    emit(template % {'to_name': to_name, 'arg_size': arg_size, 'called_name': called_name, 'called_attribute_name': called_attribute_name, 'arg_tuple': arg_tuple})
    getErrorExitCode(check_name=to_name, release_names=(called_name, called_attribute_name), needs_check=expression.mayRaiseExceptionOperation(), emit=emit, context=context)
    context.addCleanupTempName(to_name)

def _getCallCodeFromTuple(to_name, called_name, expression, args_value, emit, context):
    if False:
        print('Hello World!')
    arg_size = len(args_value)
    assert arg_size > 0
    emitLineNumberUpdateCode(expression, emit, context)
    if isMutable(args_value):
        arg_tuple_name = context.allocateTempName('call_args_kwsplit')
        arg_tuple_name.getCType().emitAssignmentCodeFromConstant(to_name=arg_tuple_name, constant=args_value, may_escape=True, emit=emit, context=context)
        args_name = arg_tuple_name
    else:
        arg_tuple_name = context.getConstantCode(constant=args_value)
        args_name = None
    quick_tuple_calls_used.add(arg_size)
    emit('%s = CALL_FUNCTION_WITH_POSARGS%d(tstate, %s, %s);\n' % (to_name, arg_size, called_name, arg_tuple_name))
    getErrorExitCode(check_name=to_name, release_names=(called_name, args_name), needs_check=expression.mayRaiseExceptionOperation(), emit=emit, context=context)
    context.addCleanupTempName(to_name)

def _getInstanceCallCodePosArgs(to_name, called_name, called_attribute_name, expression, args_name, emit, context):
    if False:
        print('Hello World!')
    emitLineNumberUpdateCode(expression, emit, context)
    emit('%s = CALL_METHOD_WITH_POSARGS(%s, %s, %s);' % (to_name, called_name, called_attribute_name, args_name))
    getErrorExitCode(check_name=to_name, release_names=(called_name, args_name), needs_check=expression.mayRaiseExceptionOperation(), emit=emit, context=context)
    context.addCleanupTempName(to_name)

def _getCallCodePosArgs(to_name, called_name, expression, args_name, emit, context):
    if False:
        i = 10
        return i + 15
    emitLineNumberUpdateCode(expression, emit, context)
    emit('%s = CALL_FUNCTION_WITH_POSARGS(tstate, %s, %s);' % (to_name, called_name, args_name))
    getErrorExitCode(check_name=to_name, release_names=(called_name, args_name), needs_check=expression.mayRaiseExceptionOperation(), emit=emit, context=context)
    context.addCleanupTempName(to_name)

def _getCallCodePosConstKeywordVariableArgs(to_name, called_name, expression, call_args, call_kw, emit, context):
    if False:
        while True:
            i = 10
    args = call_args.getCompileTimeConstant()
    kw_names = []
    dict_value_names = []
    for (count, pair) in enumerate(call_kw.subnode_pairs):
        kw_names.append(pair.getKeyCompileTimeConstant())
        dict_value_name = context.allocateTempName('kw_call_value_%d' % count)
        generateExpressionCode(to_name=dict_value_name, expression=pair.getValueNode(), emit=emit, context=context, allow_none=False)
        dict_value_names.append(dict_value_name)
    args_count = len(args)
    quick_mixed_calls_used.add((args_count, True, True))
    if isMutable(args):
        args_value_name = context.allocateTempName('call_posargs_values')
        args_value_name.getCType().emitAssignmentCodeFromConstant(to_name=args_value_name, constant=args, may_escape=True, emit=emit, context=context)
        args_name = args_value_name
    else:
        args_value_name = context.getConstantCode(args)
        args_name = None
    emitLineNumberUpdateCode(expression, emit, context)
    emit('{\n    PyObject *kw_values[%(kw_size)d] = {%(kw_values)s};\n    %(to_name)s = CALL_FUNCTION_WITH_POSARGS%(args_count)d_KWSPLIT(tstate, %(called_name)s, %(pos_args)s, kw_values, %(kw_names)s);\n}\n' % {'to_name': to_name, 'kw_values': ', '.join((str(dict_value_name) for dict_value_name in dict_value_names)), 'kw_size': len(call_kw.subnode_pairs), 'pos_args': args_value_name, 'args_count': args_count, 'called_name': called_name, 'kw_names': context.getConstantCode(tuple(kw_names))})
    getErrorExitCode(check_name=to_name, release_names=(called_name, args_name) + tuple(dict_value_names), needs_check=expression.mayRaiseExceptionOperation(), emit=emit, context=context)
    context.addCleanupTempName(to_name)

def getCallCodePosVariableKeywordVariableArgs(to_name, expression, called_name, call_args, pairs, emit, context):
    if False:
        return 10
    kw_names = []
    call_arg_names = []
    for (count, call_arg_element) in enumerate(call_args):
        call_arg_name = context.allocateTempName('kw_call_arg_value_%d' % count)
        generateExpressionCode(to_name=call_arg_name, expression=call_arg_element, emit=emit, context=context)
        call_arg_names.append(call_arg_name)
    if not pairs:
        return getCallCodePosArgsQuick(to_name=to_name, expression=expression, called_name=called_name, arg_names=call_arg_names, emit=emit, context=context)
    dict_value_names = []
    for (count, pair) in enumerate(pairs):
        kw_names.append(pair.getKeyCompileTimeConstant())
        dict_value_name = context.allocateTempName('kw_call_dict_value_%d' % count)
        generateExpressionCode(to_name=dict_value_name, expression=pair.getValueNode(), emit=emit, context=context, allow_none=False)
        dict_value_names.append(dict_value_name)
    args_count = len(call_args)
    quick_mixed_calls_used.add((args_count, False, True))
    emitLineNumberUpdateCode(expression, emit, context)
    emit('{\n    PyObject *args[] = {%(call_arg_names)s};\n    PyObject *kw_values[%(kw_size)d] = {%(kw_value_names)s};\n    %(to_name)s = CALL_FUNCTION_WITH_ARGS%(args_count)d_KWSPLIT(tstate, %(called_name)s, args, kw_values, %(kw_names)s);\n}\n' % {'to_name': to_name, 'called_name': called_name, 'call_arg_names': ', '.join((str(call_arg_name) for call_arg_name in call_arg_names)), 'kw_value_names': ', '.join((str(dict_value_name) for dict_value_name in dict_value_names)), 'kw_size': len(pairs), 'args_count': args_count, 'kw_names': context.getConstantCode(tuple(kw_names))})
    getErrorExitCode(check_name=to_name, release_names=(called_name,) + tuple(call_arg_names) + tuple(dict_value_names), needs_check=expression.mayRaiseExceptionOperation(), emit=emit, context=context)
    context.addCleanupTempName(to_name)

def _getCallCodePosConstantKeywordConstArgs(to_name, called_name, expression, call_args, call_kw, emit, context):
    if False:
        print('Hello World!')
    kw_items = tuple(call_kw.getCompileTimeConstant().items())
    args = call_args.getCompileTimeConstant()
    values = args + tuple((item[1] for item in kw_items))
    kw_names = tuple((item[0] for item in kw_items))
    arg_size = len(args)
    quick_mixed_calls_used.add((arg_size, False, False))
    if isMutable(values):
        args_values_name = context.allocateTempName('call_args_values')
        args_values_name.getCType().emitAssignmentCodeFromConstant(to_name=args_values_name, constant=values, may_escape=True, emit=emit, context=context)
        vector_name = args_values_name
    else:
        args_values_name = context.getConstantCode(values)
        vector_name = None
    emitLineNumberUpdateCode(expression, emit, context)
    emit('%s = CALL_FUNCTION_WITH_ARGS%d_VECTORCALL(tstate, %s, &PyTuple_GET_ITEM(%s, 0), %s);' % (to_name, arg_size, called_name, args_values_name, context.getConstantCode(kw_names)))
    getErrorExitCode(check_name=to_name, release_names=(called_name, vector_name), needs_check=expression.mayRaiseExceptionOperation(), emit=emit, context=context)
    context.addCleanupTempName(to_name)

def _getCallCodePosKeywordArgs(to_name, called_name, expression, call_args_name, call_kw_name, emit, context):
    if False:
        return 10
    emitLineNumberUpdateCode(expression, emit, context)
    emit('%s = CALL_FUNCTION(tstate, %s, %s, %s);' % (to_name, called_name, call_args_name, call_kw_name))
    getErrorExitCode(check_name=to_name, release_names=(called_name, call_args_name, call_kw_name), needs_check=expression.mayRaiseExceptionOperation(), emit=emit, context=context)
    context.addCleanupTempName(to_name)
max_quick_call = 10

def getQuickCallCode(args_count, has_tuple_arg):
    if False:
        while True:
            i = 10
    template = getTemplateC('nuitka.code_generation', 'CodeTemplateCallsPositional.c.j2')
    return template.render(args_count=args_count, has_tuple_arg=has_tuple_arg)

def getQuickMethodCallCode(args_count):
    if False:
        while True:
            i = 10
    template = getTemplateC('nuitka.code_generation', 'CodeTemplateCallsMethodPositional.c.j2')
    return template.render(args_count=args_count)

def getQuickMixedCallCode(args_count, has_tuple_arg, has_dict_values):
    if False:
        for i in range(10):
            print('nop')
    template = getTemplateC('nuitka.code_generation', 'CodeTemplateCallsMixed.c.j2')
    return template.render(args_count=args_count, has_tuple_arg=has_tuple_arg, has_dict_values=has_dict_values)

def getQuickMethodDescriptorCallCode(args_count):
    if False:
        print('Hello World!')
    template = getTemplateC('nuitka.code_generation', 'CodeTemplateCallsPositionalMethodDescr.c.j2')
    return template.render(args_count=args_count)

def getTemplateCodeDeclaredFunction(code):
    if False:
        return 10
    code = code.strip().split('{', 1)[0] + ';'
    return 'extern ' + code.replace(' {', ';').replace('static ', '').replace('inline ', '').replace('HEDLEY_NEVER_INLINE ', '').replace('__BINARY', 'BINARY').replace('_BINARY', 'BINARY').replace('__INPLACE', 'INPLACE').replace('_INPLACE', 'INPLACE')

def getCallsCode():
    if False:
        for i in range(10):
            print('nop')
    header_codes = []
    body_codes = []
    body_codes.append(template_helper_impl_decl % {})
    for quick_call_used in sorted(quick_calls_used.union(quick_instance_calls_used)):
        if quick_call_used <= max_quick_call:
            continue
        code = getQuickCallCode(args_count=quick_call_used, has_tuple_arg=False)
        body_codes.append(code)
        header_codes.append(getTemplateCodeDeclaredFunction(code))
    for quick_tuple_call_used in sorted(quick_tuple_calls_used):
        if quick_tuple_call_used <= max_quick_call:
            continue
        code = getQuickCallCode(args_count=quick_tuple_call_used, has_tuple_arg=True)
        body_codes.append(code)
        header_codes.append(getTemplateCodeDeclaredFunction(code))
    for (quick_mixed_call_used, has_tuple_arg, has_dict_values) in sorted(quick_mixed_calls_used):
        if quick_mixed_call_used <= max_quick_call:
            continue
        code = getQuickMixedCallCode(args_count=quick_mixed_call_used, has_tuple_arg=has_tuple_arg, has_dict_values=has_dict_values)
        body_codes.append(code)
        header_codes.append(getTemplateCodeDeclaredFunction(code))
    for quick_instance_call_used in sorted(quick_instance_calls_used):
        if quick_instance_call_used <= max_quick_call:
            continue
        code = getQuickMethodCallCode(args_count=quick_instance_call_used)
        body_codes.append(code)
        header_codes.append(getTemplateCodeDeclaredFunction(code))
    return (template_header_guard % {'header_guard_name': '__NUITKA_CALLS_H__', 'header_body': '\n'.join(header_codes)}, '\n'.join(body_codes))