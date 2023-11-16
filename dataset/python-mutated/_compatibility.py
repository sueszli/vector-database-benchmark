from typing import Any, Dict
import textwrap
_BACK_COMPAT_OBJECTS: Dict[Any, None] = {}
_MARKED_WITH_COMPATIBILITY: Dict[Any, None] = {}

def compatibility(is_backward_compatible: bool):
    if False:
        while True:
            i = 10
    if is_backward_compatible:

        def mark_back_compat(fn):
            if False:
                return 10
            docstring = textwrap.dedent(getattr(fn, '__doc__', None) or '')
            docstring += '\n.. note::\n    Backwards-compatibility for this API is guaranteed.\n'
            fn.__doc__ = docstring
            _BACK_COMPAT_OBJECTS.setdefault(fn)
            _MARKED_WITH_COMPATIBILITY.setdefault(fn)
            return fn
        return mark_back_compat
    else:

        def mark_not_back_compat(fn):
            if False:
                i = 10
                return i + 15
            docstring = textwrap.dedent(getattr(fn, '__doc__', None) or '')
            docstring += '\n.. warning::\n    This API is experimental and is *NOT* backward-compatible.\n'
            fn.__doc__ = docstring
            _MARKED_WITH_COMPATIBILITY.setdefault(fn)
            return fn
        return mark_not_back_compat