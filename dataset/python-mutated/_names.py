from nvidia.dali import fn as _functional

def _schema_name(cls):
    if False:
        i = 10
        return i + 15
    'Extract the name of the schema from Operator class.'
    return getattr(cls, 'schema_name', cls.__name__)

def _process_op_name(op_schema_name, make_hidden=False, api='ops'):
    if False:
        i = 10
        return i + 15
    'Based on the schema name (for example "Resize" or "experimental__readers__Video")\n    transform it into Python-compatible module & operator name information.\n\n    Parameters\n    ----------\n    op_schema_name : str\n        The name of the schema\n    make_hidden : bool, optional\n        Should a .hidden module be added to the module path to indicate an internal operator,\n        that it\'s later reimported but not directly discoverable, by default False\n    api : str, optional\n        API type, "ops" or "fn", by default "ops"\n\n    Returns\n    -------\n    (str, list, str)\n        (Full name with all submodules, submodule path to the operator, name of the operator),\n        for example:\n            ("Resize", [], "Resize") or\n            ("experimental.readers.Video", ["experimental", "readers"], "Video")\n    '
    namespace_delim = '__'
    op_full_name = op_schema_name.replace(namespace_delim, '.')
    (*submodule, op_name) = op_full_name.split('.')
    if make_hidden:
        submodule = [*submodule, 'hidden']
    if api == 'ops':
        return (op_full_name, submodule, op_name)
    else:
        return (op_full_name, submodule, _functional._to_snake_case(op_name))

def _op_name(op_schema_name, api='fn'):
    if False:
        i = 10
        return i + 15
    'Extract the name of the operator from the schema and return it transformed for given API:\n    CamelCase for "ops" API, and snake_case for "fn" API. The name contains full module path,\n    for example:\n        _op_name("experimental__readers__VideoResize", "fn") -> "experimental.readers.video_resize"\n\n    Parameters\n    ----------\n    op_schema_name : str\n        The name of the schema\n    api : str, optional\n        API type, "ops" or "fn", by default "fn"\n\n    Returns\n    -------\n    str\n        The fully qualified name in given API\n    '
    (full_name, submodule, op_name) = _process_op_name(op_schema_name)
    if api == 'fn':
        return '.'.join([*submodule, _functional._to_snake_case(op_name)])
    elif api == 'ops':
        return full_name
    else:
        raise ValueError(f"{api} is not a valid DALI api name, try one of {('fn', 'ops')}")