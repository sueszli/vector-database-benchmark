""" Yield related codes.

The normal "yield", and the Python 3.3 or higher "yield from" variant.
"""
from .CodeHelpers import generateChildExpressionsCode, withObjectCodeTemporaryAssignment
from .ErrorCodes import getErrorExitCode
from .PythonAPICodes import getReferenceExportCode
from .VariableDeclarations import VariableDeclaration

def _getYieldPreserveCode(to_name, value_name, preserve_exception, yield_code, resume_code, emit, context):
    if False:
        i = 10
        return i + 15
    yield_return_label = context.allocateLabel('yield_return')
    yield_return_index = yield_return_label.split('_')[-1]
    locals_preserved = context.variable_storage.getLocalPreservationDeclarations()
    if type(value_name) is tuple:
        value_names = value_name
    else:
        value_names = (value_name,)
    for name in value_names:
        if not context.needsCleanup(name):
            locals_preserved.remove(name)
    if to_name in locals_preserved:
        locals_preserved.remove(to_name)
    if locals_preserved:
        yield_tmp_storage = context.variable_storage.getVariableDeclarationTop('yield_tmps')
        if yield_tmp_storage is None:
            yield_tmp_storage = context.variable_storage.addVariableDeclarationTop('char[1024]', 'yield_tmps', None)
        emit('Nuitka_PreserveHeap(%s, %s, NULL);' % (yield_tmp_storage, ', '.join(('&%s, sizeof(%s)' % (local_preserved, local_preserved.c_type) for local_preserved in locals_preserved))))
    if preserve_exception:
        emit('SAVE_%s_EXCEPTION(tstate, %s);' % (context.getContextObjectName().upper(), context.getContextObjectName()))
    emit('%(context_object_name)s->m_yield_return_index = %(yield_return_index)s;' % {'context_object_name': context.getContextObjectName(), 'yield_return_index': yield_return_index})
    emit(yield_code)
    emit('%(yield_return_label)s:' % {'yield_return_label': yield_return_label})
    if preserve_exception:
        emit('RESTORE_%s_EXCEPTION(tstate, %s);' % (context.getContextObjectName().upper(), context.getContextObjectName()))
    if locals_preserved:
        emit('Nuitka_RestoreHeap(%s, %s, NULL);' % (yield_tmp_storage, ', '.join(('&%s, sizeof(%s)' % (local_preserved, local_preserved.c_type) for local_preserved in locals_preserved))))
    if resume_code:
        emit(resume_code)
    yield_return_name = VariableDeclaration('PyObject *', 'yield_return_value', None, None)
    getErrorExitCode(check_name=yield_return_name, emit=emit, context=context)
    emit('%s = %s;' % (to_name, yield_return_name))
    context.addCleanupTempName(to_name)

def generateYieldCode(to_name, expression, emit, context):
    if False:
        while True:
            i = 10
    (value_name,) = generateChildExpressionsCode(expression=expression, emit=emit, context=context)
    preserve_exception = expression.isExceptionPreserving()
    getReferenceExportCode(value_name, emit, context)
    if context.needsCleanup(value_name):
        context.removeCleanupTempName(value_name)
    yield_code = 'return %(yielded_value)s;' % {'yielded_value': value_name}
    with withObjectCodeTemporaryAssignment(to_name, 'yield_result', expression, emit, context) as result_name:
        _getYieldPreserveCode(to_name=result_name, value_name=value_name, yield_code=yield_code, resume_code=None, preserve_exception=preserve_exception, emit=emit, context=context)
        if to_name.c_type == 'nuitka_void':
            result_name.maybe_unused = True

def generateYieldFromCode(to_name, expression, emit, context):
    if False:
        for i in range(10):
            print('nop')
    (value_name,) = generateChildExpressionsCode(expression=expression, emit=emit, context=context)
    preserve_exception = expression.isExceptionPreserving()
    getReferenceExportCode(value_name, emit, context)
    if context.needsCleanup(value_name):
        context.removeCleanupTempName(value_name)
    yield_code = 'generator->m_yield_from = %(yield_from)s;\nreturn NULL;\n' % {'yield_from': value_name}
    with withObjectCodeTemporaryAssignment(to_name, 'yieldfrom_result', expression, emit, context) as result_name:
        _getYieldPreserveCode(to_name=result_name, value_name=value_name, yield_code=yield_code, resume_code=None, preserve_exception=preserve_exception, emit=emit, context=context)

def generateYieldFromAwaitableCode(to_name, expression, emit, context):
    if False:
        print('Hello World!')
    preserve_exception = expression.isExceptionPreserving()
    (awaited_name,) = generateChildExpressionsCode(expression=expression, emit=emit, context=context)
    yield_code = '%(object_name)s->m_yield_from = %(yield_from)s;\n%(object_name)s->m_awaiting = true;\nreturn NULL;\n' % {'object_name': context.getContextObjectName(), 'yield_from': awaited_name}
    resume_code = '%(object_name)s->m_awaiting = false;\n' % {'object_name': context.getContextObjectName()}
    getReferenceExportCode(awaited_name, emit, context)
    if context.needsCleanup(awaited_name):
        context.removeCleanupTempName(awaited_name)
    with withObjectCodeTemporaryAssignment(to_name, 'await_result', expression, emit, context) as result_name:
        _getYieldPreserveCode(to_name=result_name, value_name=awaited_name, yield_code=yield_code, resume_code=resume_code, preserve_exception=preserve_exception, emit=emit, context=context)

def getYieldReturnDispatchCode(context):
    if False:
        i = 10
        return i + 15
    function_dispatch = ['case %(index)d: goto yield_return_%(index)d;' % {'index': yield_index} for yield_index in range(context.getLabelCount('yield_return'), 0, -1)]
    if function_dispatch:
        function_dispatch.insert(0, 'switch(%s->m_yield_return_index) {' % context.getContextObjectName())
        function_dispatch.append('}')
    return function_dispatch