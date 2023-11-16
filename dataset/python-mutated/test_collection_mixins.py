import System.Collections.Generic as C

def test_contains():
    if False:
        for i in range(10):
            print('nop')
    l = C.List[int]()
    l.Add(42)
    assert 42 in l
    assert 43 not in l

def test_dict_items():
    if False:
        for i in range(10):
            print('nop')
    d = C.Dictionary[int, str]()
    d[42] = 'a'
    items = d.items()
    assert len(items) == 1
    (k, v) = items[0]
    assert k == 42
    assert v == 'a'

def test_dict_in_keys():
    if False:
        for i in range(10):
            print('nop')
    d = C.Dictionary[str, int]()
    d['a'] = 42
    assert 'a' in d.Keys
    assert 'b' not in d.Keys