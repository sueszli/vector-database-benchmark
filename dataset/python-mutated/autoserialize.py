import inspect

def _is_marked_autoserializable(object):
    if False:
        for i in range(10):
            print('nop')
    return getattr(object, '_is_autoserialize', False)

def _discover_autoserialize(module, visited):
    if False:
        while True:
            i = 10
    '\n    Traverses a module tree given by the head module and\n    returns all functions that are marked with ``@autoserialize`` decorator.\n\n    :param module: Module currently searched.\n    :param visited: Paths to the ``__init__.py`` of the modules already searched.\n    :return: All functions that are marked with ``@autoserialize`` decorator.\n    '
    assert module is not None
    ret = []
    try:
        module_members = inspect.getmembers(module)
    except (ModuleNotFoundError, ImportError):
        return ret
    modules = []
    for (name, path) in module_members:
        obj = getattr(module, name, None)
        if inspect.ismodule(obj) and path not in visited:
            modules.append(name)
            visited.append(path)
        elif inspect.isfunction(obj) and _is_marked_autoserializable(obj):
            ret.append(obj)
    for mod in modules:
        ret.extend(_discover_autoserialize(getattr(module, mod, None), visited=visited))
    return ret

def invoke_autoserialize(head_module, filename):
    if False:
        print('Hello World!')
    '\n    Perform the autoserialization of a function marked by\n        :meth:`nvidia.dali.plugin.triton.autoserialize`.\n\n    Assuming, that user marked a function with ``@autoserialize`` decorator, the\n    ``invoke_autoserialize`` is a utility function, which will actually perform\n    the autoserialization.\n    It discovers the ``@autoserialize`` function in a module tree denoted by provided\n    ``head_module`` and saves the serialized DALI pipeline to the file in the ``filename`` path.\n\n    Only one ``@autoserialize`` function may exist in a given module tree.\n\n    :param head_module: Module, denoting the model tree in which the decorated function shall exist.\n    :param filename: Path to the file, where the output of serialization will be saved.\n    '
    autoserialize_functions = _discover_autoserialize(head_module, visited=[])
    if len(autoserialize_functions) > 1:
        raise RuntimeError(f'Precisely one autoserialize function must exist in the module. Found {len(autoserialize_functions)}: {autoserialize_functions}.')
    if len(autoserialize_functions) < 1:
        raise RuntimeError('Precisely one autoserialize function must exist in the module. Found none.')
    dali_pipeline = autoserialize_functions[0]
    dali_pipeline().serialize(filename=filename)