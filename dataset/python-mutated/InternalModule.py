""" Internal module

This is a container for helper functions that are shared across modules. It
may not exist, and is treated specially in code generation. This avoids to
own these functions to a random module.

TODO: Clarify by renaming that the top module is now used, and these are
merely helpers to do it.
"""
from nuitka.ModuleRegistry import getRootTopModule
from nuitka.nodes.FunctionNodes import ExpressionFunctionPureBody, ExpressionFunctionPureInlineConstBody
from nuitka.SourceCodeReferences import fromFilename
internal_source_ref = fromFilename('internal').atInternal()

def once_decorator(func):
    if False:
        return 10
    "Cache result of a function call without arguments.\n\n    Used for all internal function accesses to become a singleton.\n\n    Note: This doesn't much specific anymore, but we are not having\n    this often enough to warrant reuse or generalization.\n\n    "
    func.cached_value = None

    def replacement():
        if False:
            return 10
        if func.cached_value is None:
            func.cached_value = func()
        return func.cached_value
    return replacement

@once_decorator
def getInternalModule():
    if False:
        i = 10
        return i + 15
    'Get the singleton internal module.'
    return getRootTopModule()

def makeInternalHelperFunctionBody(name, parameters, inline_const_args=False):
    if False:
        for i in range(10):
            print('nop')
    if inline_const_args:
        node_class = ExpressionFunctionPureInlineConstBody
    else:
        node_class = ExpressionFunctionPureBody
    result = node_class(provider=getInternalModule(), name=name, code_object=None, doc=None, parameters=parameters, flags=None, auto_release=None, source_ref=internal_source_ref)
    for variable in parameters.getAllVariables():
        result.removeVariableReleases(variable)
    return result