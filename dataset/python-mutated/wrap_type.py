from types import BuiltinMethodType, FunctionType, GetSetDescriptorType, MethodDescriptorType, WrapperDescriptorType
from functorch._C import dim as _C
_wrap_method = _C._wrap_method
FUNC_TYPES = (FunctionType, MethodDescriptorType, BuiltinMethodType, WrapperDescriptorType)
PROPERTY_TYPES = (GetSetDescriptorType, property)

def _py_wrap_method(orig, __torch_function__):
    if False:
        i = 10
        return i + 15

    def impl(*args, **kwargs):
        if False:
            return 10
        return __torch_function__(orig, None, args, kwargs)
    return impl

def wrap_type(use_c, to_patch, pattern, __torch_function__):
    if False:
        while True:
            i = 10
    if use_c:
        wrap_method = _wrap_method
    else:
        wrap_method = _py_wrap_method
    all = {}
    for t in reversed(pattern.mro()[:-1]):
        all.update(t.__dict__)

    def wrap_attr(orig):
        if False:
            return 10
        return property(wrap_method(orig.__get__, __torch_function__))
    for (name, obj) in all.items():
        if name in ('__dict__', '__new__', '__init__', '__repr__', '__weakref__', '__doc__', '__module__', '__dir__'):
            continue
        if hasattr(to_patch, name) and getattr(to_patch, name) is not getattr(object, name, None):
            continue
        if isinstance(obj, FUNC_TYPES):
            setattr(to_patch, name, wrap_method(obj, __torch_function__))
        elif isinstance(obj, PROPERTY_TYPES):
            setattr(to_patch, name, wrap_attr(obj))