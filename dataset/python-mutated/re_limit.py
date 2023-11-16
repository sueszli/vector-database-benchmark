try:
    import re
except ImportError:
    print('SKIP')
    raise SystemExit

def test_re(r):
    if False:
        return 10
    try:
        re.compile(r)
    except:
        print('Error')
test_re('[' + 'a' * 256 + ']')
test_re('(a)' * 256)
test_re('(' + 'a' * 62 + ')?')
test_re('(' + 'a' * 60 + '.)*')
test_re('(' + 'a' * 60 + '..)*')
test_re('(' + 'a' * 62 + ')+')
test_re('b' * 63 + '|a')