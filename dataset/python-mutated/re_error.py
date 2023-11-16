try:
    import re
except ImportError:
    print('SKIP')
    raise SystemExit

def test_re(r):
    if False:
        while True:
            i = 10
    try:
        re.compile(r)
        print('OK')
    except:
        print('Error')
test_re('?')
test_re('*')
test_re('+')
test_re(')')
test_re('[')
test_re('([')
test_re('([)')
test_re('[a\\]')