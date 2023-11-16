if a or True:
    pass
if (a or b) or True:
    pass
if a or (b or True):
    pass
if a and True:
    pass
if True:
    pass

def validate(self, value):
    if False:
        return 10
    return json.loads(value) or True
if a or f() or b or g() or True:
    pass
if a or f() or True or g() or b:
    pass
if True or f() or a or g() or b:
    pass
if a or True or f() or b or g():
    pass
if a and f() and b and g() and False:
    pass
if a and f() and False and g() and b:
    pass
if False and f() and a and g() and b:
    pass
if a and False and f() and b and g():
    pass
a or '' or True
a or 'foo' or True or 'bar'
a or 0 or True
a or 1 or True or 2
a or 0.0 or True
a or 0.1 or True or 0.2
a or [] or True
a or list([]) or True
a or [1] or True or [2]
a or list([1]) or True or list([2])
a or {} or True
a or dict() or True
a or {1: 1} or True or {2: 2}
a or dict({1: 1}) or True or dict({2: 2})
a or set() or True
a or set(set()) or True
a or {1} or True or {2}
a or set({1}) or True or set({2})
a or () or True
a or tuple(()) or True
a or (1,) or True or (2,)
a or tuple((1,)) or True or tuple((2,))
a or frozenset() or True
a or frozenset(frozenset()) or True
a or frozenset({1}) or True or frozenset({2})
a or frozenset(frozenset({1})) or True or frozenset(frozenset({2}))
bool(a or [1] or True or [2])
assert a or [1] or True or [2]
if (a or [1] or True or [2]) and (a or [1] or True or [2]):
    pass
0 if a or [1] or True or [2] else 1
while a or [1] or True or [2]:
    pass
[0 for a in range(10) for b in range(10) if a or [1] or True or [2] if b or [1] or True or [2]]
{0 for a in range(10) for b in range(10) if a or [1] or True or [2] if b or [1] or True or [2]}
{0: 0 for a in range(10) for b in range(10) if a or [1] or True or [2] if b or [1] or True or [2]}
(0 for a in range(10) for b in range(10) if a or [1] or True or [2] if b or [1] or True or [2])
a or [1] or True or [2]
if (a or [1] or True or [2]) == (a or [1]):
    pass
if f(a or [1] or True or [2]):
    pass

def secondToTime(s0: int) -> (int, int, int) or str:
    if False:
        i = 10
        return i + 15
    (m, s) = divmod(s0, 60)

def secondToTime(s0: int) -> (int, int, int) or str:
    if False:
        while True:
            i = 10
    (m, s) = divmod(s0, 60)