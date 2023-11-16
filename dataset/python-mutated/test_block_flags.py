from grc.core.blocks._flags import Flags

def test_simple_init():
    if False:
        print('Hello World!')
    assert 'test' not in Flags()
    assert 'test' in Flags(' test')
    assert 'test' in Flags('test, foo')
    assert 'test' in Flags({'test', 'foo'})

def test_deprecated():
    if False:
        for i in range(10):
            print('nop')
    assert Flags.DEPRECATED == 'deprecated'
    assert Flags('this is deprecated').deprecated is True

def test_extend():
    if False:
        print('Hello World!')
    f = Flags('a')
    f.set('b')
    assert isinstance(f, Flags)
    f.set(u'b')
    assert isinstance(f, Flags)
    f = Flags(u'a')
    f.set('b')
    assert isinstance(f, Flags)
    f.set(u'b')
    assert isinstance(f, Flags)
    assert f.data == {'a', 'b'}