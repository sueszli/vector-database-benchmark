import types
from typing import Union

def get_method_funcs(cls_def_str: str, obj):
    if False:
        while True:
            i = 10
    '\n    Parses the class definition string and returns a dict with python functions under their names,\n    parsed from the methods in the class definition string using the ast module.\n    '
    import sys
    if sys.version_info < (3, 9):
        raise Exception('src code modifications are only supported on Python >=3.9')
    import ast
    ast_funcs: [ast.FunctionDef] = [f for f in ast.parse(cls_def_str).body[0].body if type(f) == ast.FunctionDef]
    funcs = {}
    for astf in ast_funcs:
        d = __builtins__.copy()
        exec(ast.unparse(astf), d)
        f = d[astf.name]
        funcs[astf.name] = f
    return funcs

class SrcCodeUpdater:
    """
    Provides functionality to override method implementations of an inspectable object.
    """

    @staticmethod
    def override_code(obj: object, new_class_src) -> Union[None, Exception]:
        if False:
            while True:
                i = 10
        try:
            funcs = get_method_funcs(new_class_src, obj)
            for (name, f) in funcs.items():
                setattr(obj, name, types.MethodType(f, obj))
        except Exception as e:
            return e

    @staticmethod
    def override_code_OLD(obj: object, orig_class_src, orig_mod_src, new_class_src):
        if False:
            for i in range(10):
                print('nop')
        import inspect
        new_module_code = orig_mod_src.replace(orig_class_src, new_class_src)
        module = types.ModuleType('new_class_module')
        module.__file__ = inspect.getfile(obj.__class__)
        exec(new_module_code, module.__dict__)
        new_obj_class = getattr(module, type(obj).__name__)
        f_methods = inspect.getmembers(new_obj_class, predicate=inspect.ismethod)
        functions = inspect.getmembers(new_obj_class, predicate=inspect.isfunction)
        for (m_name, m_obj) in f_methods + functions:
            setattr(obj, m_name, types.MethodType(m_obj, obj))