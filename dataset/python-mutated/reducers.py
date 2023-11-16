import inspect
import pickle
import sys
import types
import marshal
import importlib

def dummy_lambda():
    if False:
        while True:
            i = 10
    pass

def get_global_references_from_nested_code(code, global_scope, global_refs):
    if False:
        while True:
            i = 10
    for constant in code.co_consts:
        if inspect.iscode(constant):
            closure = tuple((types.CellType(None) for _ in range(len(constant.co_freevars))))
            dummy_function = types.FunctionType(constant, global_scope, 'dummy_function', closure=closure)
            global_refs.update(inspect.getclosurevars(dummy_function).globals)
            get_global_references_from_nested_code(constant, global_scope, global_refs)

def set_funcion_state(fun, state):
    if False:
        return 10
    fun.__globals__.update(state['global_refs'])
    fun.__defaults__ = state['defaults']
    fun.__kwdefaults__ = state['kwdefaults']

def function_unpickle(name, qualname, code, closure):
    if False:
        return 10
    code = marshal.loads(code)
    global_scope = {'__builtins__': __builtins__}
    fun = types.FunctionType(code, global_scope, name, closure=closure)
    fun.__qualname__ = qualname
    return fun

def function_by_value_reducer(fun):
    if False:
        for i in range(10):
            print('nop')
    cl_vars = inspect.getclosurevars(fun)
    code = marshal.dumps(fun.__code__)
    basic_def = (fun.__name__, fun.__qualname__, code, fun.__closure__)
    global_refs = dict(cl_vars.globals)
    get_global_references_from_nested_code(fun.__code__, fun.__globals__, global_refs)
    fun_context = {'global_refs': global_refs, 'defaults': fun.__defaults__, 'kwdefaults': fun.__kwdefaults__}
    return (function_unpickle, basic_def, fun_context, None, None, set_funcion_state)

def module_unpickle(name, origin, submodule_search_locations):
    if False:
        for i in range(10):
            print('nop')
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, origin, submodule_search_locations=submodule_search_locations)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def module_reducer(module):
    if False:
        print('Hello World!')
    spec = module.__spec__
    return (module_unpickle, (spec.name, spec.origin, spec.submodule_search_locations))

def set_cell_state(cell, state):
    if False:
        return 10
    cell.cell_contents = state['cell_contents']

def cell_unpickle():
    if False:
        i = 10
        return i + 15
    return types.CellType(None)

def cell_reducer(cell):
    if False:
        while True:
            i = 10
    return (cell_unpickle, tuple(), {'cell_contents': cell.cell_contents}, None, None, set_cell_state)

class DaliCallbackPickler(pickle.Pickler):

    def reducer_override(self, obj):
        if False:
            i = 10
            return i + 15
        if inspect.ismodule(obj):
            return module_reducer(obj)
        if isinstance(obj, types.CellType):
            return cell_reducer(obj)
        if inspect.isfunction(obj):
            if isinstance(obj, type(dummy_lambda)) and obj.__name__ == dummy_lambda.__name__ or getattr(obj, '_dali_pickle_by_value', False):
                return function_by_value_reducer(obj)
            try:
                pickle.dumps(obj)
            except AttributeError as e:
                if "Can't pickle local object" in str(e):
                    return function_by_value_reducer(obj)
            except pickle.PicklingError as e:
                if "it's not the same object as" in str(e):
                    return function_by_value_reducer(obj)
        return NotImplemented