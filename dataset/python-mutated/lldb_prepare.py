def load_lib_and_attach(debugger, command, result, internal_dict):
    if False:
        return 10
    import shlex
    args = shlex.split(command)
    dll = args[0]
    is_debug = args[1]
    python_code = args[2]
    show_debug_info = args[3]
    import lldb
    options = lldb.SBExpressionOptions()
    options.SetFetchDynamicValue()
    options.SetTryAllThreads(run_others=False)
    options.SetTimeoutInMicroSeconds(timeout=60000000)
    print(dll)
    target = debugger.GetSelectedTarget()
    res = target.EvaluateExpression('(void*)dlopen("%s", 2);' % dll, options)
    error = res.GetError()
    if error:
        print(error)
    print(python_code)
    res = target.EvaluateExpression('(int)DoAttach(%s, "%s", %s);' % (is_debug, python_code.replace('"', "'"), show_debug_info), options)
    error = res.GetError()
    if error:
        print(error)

def __lldb_init_module(debugger, internal_dict):
    if False:
        return 10
    import lldb
    debugger.HandleCommand('command script add -f lldb_prepare.load_lib_and_attach load_lib_and_attach')
    try:
        target = debugger.GetSelectedTarget()
        if target:
            process = target.GetProcess()
            if process:
                for thread in process:
                    internal_dict['_thread_%d' % thread.GetThreadID()] = True
    except:
        import traceback
        traceback.print_exc()