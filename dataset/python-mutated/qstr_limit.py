def make_id(n, base='a'):
    if False:
        while True:
            i = 10
    return ''.join((chr(ord(base) + i % 26) for i in range(n)))
for l in range(254, 259):
    g = {}
    var = make_id(l)
    try:
        exec(var + '=1', g)
    except RuntimeError:
        print('RuntimeError', l)
        continue
    print(var in g)

def f(**k):
    if False:
        while True:
            i = 10
    print(k)
for l in range(254, 259):
    try:
        exec('f({}=1)'.format(make_id(l)))
    except RuntimeError:
        print('RuntimeError', l)
for l in range(254, 259):
    id = make_id(l)
    try:
        print(type(id, (), {}).__name__)
    except RuntimeError:
        print('RuntimeError', l)

class A:
    pass
for l in range(254, 259):
    id = make_id(l)
    a = A()
    try:
        setattr(a, id, 123)
    except RuntimeError:
        print('RuntimeError', l)
    try:
        print(hasattr(a, id), getattr(a, id))
    except RuntimeError:
        print('RuntimeError', l)
for l in range(254, 259):
    id = make_id(l)
    try:
        print(('{' + id + '}').format(**{id: l}))
    except RuntimeError:
        print('RuntimeError', l)
for l in range(254, 259):
    id = make_id(l)
    try:
        print(('%(' + id + ')d') % {id: l})
    except RuntimeError:
        print('RuntimeError', l)
for l in (100, 101, 256, 257, 258):
    try:
        __import__(make_id(l))
    except ImportError:
        print('ok', l)
    except RuntimeError:
        print('RuntimeError', l)
for l in (100, 101, 102, 128, 129):
    try:
        exec('import ' + make_id(l) + '.' + make_id(l, 'A'))
    except ImportError:
        print('ok', l)
    except RuntimeError:
        print('RuntimeError', l)