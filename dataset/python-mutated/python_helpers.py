def test1(l):
    if False:
        for i in range(10):
            print('nop')
    return ['foo', 'bar']

def test2(l, l2):
    if False:
        return 10
    return [l[0], l2[0]]