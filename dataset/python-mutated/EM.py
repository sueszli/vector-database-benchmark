from __future__ import annotations

def f_a():
    if False:
        while True:
            i = 10
    raise RuntimeError('This is an example exception')

def f_a_short():
    if False:
        i = 10
        return i + 15
    raise RuntimeError('Error')

def f_a_empty():
    if False:
        i = 10
        return i + 15
    raise RuntimeError('')

def f_b():
    if False:
        for i in range(10):
            print('nop')
    example = 'example'
    raise RuntimeError(f'This is an {example} exception')

def f_c():
    if False:
        while True:
            i = 10
    raise RuntimeError('This is an {example} exception'.format(example='example'))

def f_ok():
    if False:
        while True:
            i = 10
    msg = 'hello'
    raise RuntimeError(msg)

def f_unfixable():
    if False:
        print('Hello World!')
    msg = 'hello'
    raise RuntimeError('This is an example exception')

def f_msg_in_nested_scope():
    if False:
        return 10

    def nested():
        if False:
            for i in range(10):
                print('nop')
        msg = 'hello'
    raise RuntimeError('This is an example exception')

def f_msg_in_parent_scope():
    if False:
        for i in range(10):
            print('nop')
    msg = 'hello'

    def nested():
        if False:
            i = 10
            return i + 15
        raise RuntimeError('This is an example exception')

def f_fix_indentation_check(foo):
    if False:
        return 10
    if foo:
        raise RuntimeError('This is an example exception')
    elif foo == 'foo':
        raise RuntimeError(f'This is an exception: {foo}')
    raise RuntimeError('This is an exception: {}'.format(foo))
if foo:
    raise RuntimeError('This is an example exception')
if foo:
    x = 1
    raise RuntimeError('This is an example exception')

def f_triple_quoted_string():
    if False:
        print('Hello World!')
    raise RuntimeError(f"This is an {'example'} exception")

def f_multi_line_string():
    if False:
        return 10
    raise RuntimeError('firstsecond')

def f_multi_line_string2():
    if False:
        print('Hello World!')
    raise RuntimeError('This is an {example} exception'.format(example='example'))

def f_multi_line_string2():
    if False:
        for i in range(10):
            print('nop')
    raise RuntimeError('This is an {example} exception'.format(example='example'))