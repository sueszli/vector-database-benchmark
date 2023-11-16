try:
    {} | {}
except TypeError:
    print('SKIP')
    raise SystemExit

def print_sorted_dict(d):
    if False:
        i = 10
        return i + 15
    print(sorted(d.items()))

def test_union(a, b):
    if False:
        return 10
    print_sorted_dict(a | b)
    print_sorted_dict(b | a)
    a |= a
    print_sorted_dict(a)
    a |= b
    print_sorted_dict(a)
d = {}
e = {}
test_union(d, e)
d = {1: 'apple'}
e = {1: 'cheese'}
test_union(d, e)
d = {'spam': 1, 'eggs': 2, 'cheese': 3}
e = {'cheese': 'cheddar', 'aardvark': 'Ethel'}
test_union(d, e)