bla = b'bla'
d = {'a': b'bla', 'b': b'bla', 'c': b'bla'}

def foo(one_arg):
    if False:
        print('Hello World!')
    pass
f'{str(bla)}, {repr(bla)}, {ascii(bla)}'
f"{str(d['a'])}, {repr(d['b'])}, {ascii(d['c'])}"
f'{str(bla)}, {repr(bla)}, {ascii(bla)}'
f'{bla!s}, {repr(bla)}, {ascii(bla)}'
f'{foo(bla)}'
f"{str(bla, 'ascii')}, {str(bla, encoding='cp1255')}"
f"{bla!s} {[]!r} {'bar'!a}"
'Not an f-string {str(bla)}, {repr(bla)}, {ascii(bla)}'

def ascii(arg):
    if False:
        return 10
    pass
f'{ascii(bla)}'
f'Member of tuple mismatches type at index {i}. Expected {of_shape_i}. Got  intermediary content  that flows {repr(obj)} of type {type(obj)}.{additional_message}'
f'{str({})}'