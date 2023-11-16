from collections import namedtuple
from pytest import raises
from boltons.debugutils import wrap_trace

def test_trace_dict():
    if False:
        i = 10
        return i + 15
    target = {}
    wrapped = wrap_trace(target)
    assert target is not wrapped
    assert isinstance(wrapped, dict)
    wrapped['a'] = 'A'
    assert target['a'] == 'A'
    assert len(wrapped) == len(target)
    wrapped.pop('a')
    assert 'a' not in target
    with raises(AttributeError):
        wrapped.nonexistent_attr = 'nope'
    return

def test_trace_bytes():
    if False:
        i = 10
        return i + 15
    target = u'Hello'.encode('ascii')
    wrapped = wrap_trace(target)
    assert target is not wrapped
    assert isinstance(wrapped, bytes)
    assert len(wrapped) == len(target)
    assert wrapped.decode('utf-8') == u'Hello'
    assert wrapped.lower() == target.lower()

def test_trace_exc():
    if False:
        print('Hello World!')

    class TestException(Exception):
        pass
    target = TestException('exceptions can be a good thing')
    wrapped = wrap_trace(target)
    try:
        raise wrapped
    except TestException as te:
        assert te.args == target.args

def test_trace_which():
    if False:
        return 10

    class Config(object):

        def __init__(self, value):
            if False:
                return 10
            self.value = value
    config = Config('first')
    wrapped = wrap_trace(config, which='__setattr__')
    wrapped.value = 'second'
    assert config.value == 'second'

def test_trace_namedtuple():
    if False:
        while True:
            i = 10
    TargetType = namedtuple('TargetType', 'x y z')
    target = TargetType(1, 2, 3)
    wrapped = wrap_trace(target)
    assert wrapped == (1, 2, 3)

def test_trace_oldstyle():
    if False:
        print('Hello World!')

    class Oldie:
        test = object()

        def get_test(self):
            if False:
                print('Hello World!')
            return self.test
    oldie = Oldie()
    wrapped = wrap_trace(oldie)
    assert wrapped.get_test() is oldie.test
    return