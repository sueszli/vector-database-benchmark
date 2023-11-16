import inspect
import sys

def make_decorator(test_case_generator):
    if False:
        i = 10
        return i + 15

    def f(cls):
        if False:
            for i in range(10):
                print('nop')
        module_name = cls.__module__
        module = sys.modules[module_name]
        assert module.__name__ == module_name
        for (cls_name, members, method_generator) in test_case_generator(cls):
            _generate_case(cls, module, cls_name, members, method_generator)
        return None
    return f

def _generate_case(base, module, cls_name, mb, method_generator):
    if False:
        for i in range(10):
            print('nop')
    members = mb.copy()
    base_methods = inspect.getmembers(base, predicate=inspect.isfunction)
    for (name, value) in base_methods:
        if not name.startswith('test_'):
            continue
        value = method_generator(value)
        members[name] = value
    cls = type(cls_name, (base,), members)
    cls.__module__ = module.__name__
    setattr(module, cls_name, cls)