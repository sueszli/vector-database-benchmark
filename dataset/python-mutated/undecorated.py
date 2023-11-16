from inspect import isfunction, ismethod, isclass
__version__ = '0.3.0'

def undecorated(o):
    if False:
        return 10
    'Remove all decorators from a function, method or class'
    if isinstance(o, type):
        return o
    try:
        closure = o.func_closure
    except AttributeError:
        pass
    try:
        closure = o.__closure__
    except AttributeError:
        return
    if closure:
        for cell in closure:
            if cell.cell_contents is o:
                continue
            if looks_like_a_decorator(cell.cell_contents):
                undecd = undecorated(cell.cell_contents)
                if undecd:
                    return undecd
    return o

def looks_like_a_decorator(a):
    if False:
        print('Hello World!')
    return isfunction(a) or ismethod(a) or isclass(a)