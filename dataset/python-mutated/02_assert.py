def __call__(arg, dest):
    if False:
        return 10
    try:
        assert arg == 'spam', 'dest: %s' % dest
    except:
        raise
__call__('spam', __file__)

def refactor_doctest(clipped, new):
    if False:
        print('Hello World!')
    assert clipped, clipped
    if not new:
        new += u'\n'
    return

def test_threaded_hashing():
    if False:
        for i in range(10):
            print('nop')
    for threadnum in xrange(1):
        result = 1
        assert result > 0
        result = 2
    return result
assert test_threaded_hashing() == 2