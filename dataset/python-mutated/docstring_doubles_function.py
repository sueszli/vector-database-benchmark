def foo():
    if False:
        return 10
    'function without params, single line docstring'
    ' not a docstring'
    return

def foo2():
    if False:
        i = 10
        return i + 15
    '\n        function without params, multiline docstring\n    '
    ' not a docstring'
    return

def fun_with_params_no_docstring(a, b='\n    not a\ndocstring'):
    if False:
        return 10
    pass

def fun_with_params_no_docstring2(a, b=c[foo():], c=' not a docstring '):
    if False:
        i = 10
        return i + 15
    pass

def function_with_single_docstring(a):
    if False:
        while True:
            i = 10
    'Single line docstring'

def double_inside_single(a):
    if False:
        for i in range(10):
            print('nop')
    'Double inside "single "'