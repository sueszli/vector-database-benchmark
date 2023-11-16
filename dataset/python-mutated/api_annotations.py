from typing import Optional

def PublicAPI(*args, **kwargs):
    if False:
        print('Hello World!')
    'Annotation for documenting public APIs. Public APIs are classes and methods exposed to end users of Ludwig.\n\n    If stability="stable", the APIs will remain backwards compatible across minor Ludwig releases\n    (e.g., Ludwig 0.6 -> Ludwig 0.7).\n\n    If stability="experimental", the APIs can be used by advanced users who are tolerant to and expect\n    breaking changes. This will likely be seen in the case of incremental new feature development.\n\n    Args:\n        stability: One of {"stable", "experimental"}\n\n    Examples:\n        >>> from api_annotations import PublicAPI\n        >>> @PublicAPI\n        ... def func1(x):\n        ...     return x\n        >>> @PublicAPI(stability="experimental")\n        ... def func2(y):\n        ...     return y\n    '
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        return PublicAPI(stability='stable')(args[0])
    if 'stability' in kwargs:
        stability = kwargs['stability']
        assert stability in ['stable', 'experimental'], stability
    elif kwargs:
        raise ValueError(f'Unknown kwargs: {kwargs.keys()}')
    else:
        stability = 'stable'

    def wrap(obj):
        if False:
            return 10
        if stability == 'experimental':
            message = f'PublicAPI ({stability}): This API is {stability} and may change before becoming stable.'
        else:
            message = 'PublicAPI: This API is stable across Ludwig releases.'
        _append_doc(obj, message=message)
        _mark_annotated(obj)
        return obj
    return wrap

def DeveloperAPI(*args, **kwargs):
    if False:
        i = 10
        return i + 15
    'Annotation for documenting developer APIs. Developer APIs are lower-level methods explicitly exposed to\n    advanced Ludwig users and library developers. Their interfaces may change across minor Ludwig releases (for\n    e.g., Ludwig 0.6.1 and Ludwig 0.6.2).\n\n    Examples:\n        >>> from api_annotations import DeveloperAPI\n        >>> @DeveloperAPI\n        ... def func(x):\n        ...     return x\n    '
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        return DeveloperAPI()(args[0])

    def wrap(obj):
        if False:
            i = 10
            return i + 15
        _append_doc(obj, message='DeveloperAPI: This API may change across minor Ludwig releases.')
        _mark_annotated(obj)
        return obj
    return wrap

def Deprecated(*args, **kwargs):
    if False:
        return 10
    'Annotation for documenting a deprecated API. Deprecated APIs may be removed in future releases of Ludwig\n    (e.g., Ludwig 0.7 to Ludwig 0.8).\n\n    Args:\n        message: A message to help users understand the reason for the deprecation, and provide a migration path.\n\n    Examples:\n        >>> from api_annotations import Deprecated\n        >>> @Deprecated\n        ... def func(x):\n        ...     return x\n        >>> @Deprecated(message="g() is deprecated because the API is error prone. Please call h() instead.")\n        ... def g(y):\n        ...     return y\n    '
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        return Deprecated()(args[0])
    message = '**DEPRECATED:** This API is deprecated and may be removed in a future Ludwig release.'
    if 'message' in kwargs:
        message += ' ' + kwargs['message']
        del kwargs['message']
    if kwargs:
        raise ValueError(f'Unknown kwargs: {kwargs.keys()}')

    def inner(obj):
        if False:
            return 10
        _append_doc(obj, message=message, directive='warning')
        _mark_annotated(obj)
        return obj
    return inner

def _append_doc(obj, message: str, directive: Optional[str]=None) -> str:
    if False:
        return 10
    "\n    Args:\n        message: An additional message to append to the end of docstring for a class\n                 or method that uses one of the API annotations\n        directive: A shorter message that provides contexts for the message and indents it.\n                For example, this could be something like 'warning' or 'info'.\n    "
    if not obj.__doc__:
        obj.__doc__ = ''
    obj.__doc__ = obj.__doc__.rstrip()
    indent = _get_indent(obj.__doc__)
    obj.__doc__ += '\n\n'
    if directive is not None:
        obj.__doc__ += f"{' ' * indent}.. {directive}::\n"
        obj.__doc__ += f"{' ' * (indent + 4)}{message}"
    else:
        obj.__doc__ += f"{' ' * indent}{message}"
    obj.__doc__ += f"\n{' ' * indent}"

def _mark_annotated(obj) -> None:
    if False:
        for i in range(10):
            print('nop')
    if hasattr(obj, '__name__'):
        obj._annotated = obj.__name__

def _is_annotated(obj) -> bool:
    if False:
        while True:
            i = 10
    return hasattr(obj, '_annotated') and obj._annotated == obj.__name__

def _get_indent(docstring: str) -> int:
    if False:
        while True:
            i = 10
    "\n    Example:\n        >>> def f():\n        ...     '''Docstring summary.'''\n        >>> f.__doc__\n        'Docstring summary.'\n        >>> _get_indent(f.__doc__)\n        0\n        >>> def g(foo):\n        ...     '''Docstring summary.\n        ...\n        ...     Args:\n        ...         foo: Does bar.\n        ...     '''\n        >>> g.__doc__\n        'Docstring summary.\\n\\n    Args:\\n        foo: Does bar.\\n    '\n        >>> _get_indent(g.__doc__)\n        4\n        >>> class A:\n        ...     def h():\n        ...         '''Docstring summary.\n        ...\n        ...         Returns:\n        ...             None.\n        ...         '''\n        >>> A.h.__doc__\n        'Docstring summary.\\n\\n        Returns:\\n            None.\\n        '\n        >>> _get_indent(A.h.__doc__)\n        8\n    "
    if not docstring:
        return 0
    non_empty_lines = list(filter(bool, docstring.splitlines()))
    if len(non_empty_lines) == 1:
        return 0
    return len(non_empty_lines[1]) - len(non_empty_lines[1].lstrip())