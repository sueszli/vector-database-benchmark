def type_to_string(_type: type) -> str:
    if False:
        while True:
            i = 10
    'Gets the string representation of a type.\n\n    THe original type can be derived from the returned string representation through\n    pydoc.locate().\n    '
    if _type.__module__ == 'typing':
        return f'{_type.__module__}.{_type._name}'
    elif _type.__module__ == 'builtins':
        return _type.__name__
    else:
        return f'{_type.__module__}.{_type.__name__}'