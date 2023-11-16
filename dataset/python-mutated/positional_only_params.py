def positional_only_call(a, /, b):
    if False:
        for i in range(10):
            print('nop')
    a
    b
    if UNDEFINED:
        return a
    else:
        return b
positional_only_call('', 1)

def positional_only_call2(a, /, b=3):
    if False:
        for i in range(10):
            print('nop')
    if UNDEFINED:
        return a
    else:
        return b
positional_only_call2(1)
positional_only_call2(SOMETHING_UNDEFINED)
positional_only_call2(SOMETHING_UNDEFINED, '')
positional_only_call2(a=1, b='')
positional_only_call2(b='', a=tuple)