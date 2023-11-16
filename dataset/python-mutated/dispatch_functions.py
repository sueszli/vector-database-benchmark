from ...utils import BreakGraphError, FallbackError

def raise_break_graph_fn(*args, **kwarg):
    if False:
        print('Hello World!')
    raise BreakGraphError('raise by raise_break_graph_fn.')

def raise_not_implement_fn(*args, **kwarg):
    if False:
        i = 10
        return i + 15
    raise FallbackError('raise by raise_break_graph_fn.')

def operator_in(left, right):
    if False:
        print('Hello World!')
    return left in right

def operator_not_in(left, right):
    if False:
        print('Hello World!')
    return left not in right

def operator_exception_match(left, right):
    if False:
        return 10
    pass

def operator_BAD(left, right):
    if False:
        i = 10
        return i + 15
    pass

def operator_is_none(val):
    if False:
        return 10
    pass

def operator_is_not_none(val):
    if False:
        for i in range(10):
            print('nop')
    pass

def tensor_numel(x):
    if False:
        return 10
    pass