def _format(node):
    if False:
        i = 10
        return i + 15
    return [(a, int(b)) for (a, b) in node.items()]

def monthrange(ary, dotext):
    if False:
        while True:
            i = 10
    return [a[3:] for a in ary if a.startswith(dotext)]
x = {'a': '1', 'b': '2'}
assert [('a', 1), ('b', 2)] == _format(x)
ary = ['Monday', 'Twoday', 'Monmonth']
assert ['day', 'month'] == monthrange(ary, 'Mon')