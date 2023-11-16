from __future__ import annotations
from ansible.utils.unsafe_proxy import AnsibleUnsafe, AnsibleUnsafeBytes, AnsibleUnsafeText, wrap_var
from ansible.module_utils.common.text.converters import to_text, to_bytes

def test_wrap_var_text():
    if False:
        return 10
    assert isinstance(wrap_var(u'foo'), AnsibleUnsafeText)

def test_wrap_var_bytes():
    if False:
        for i in range(10):
            print('nop')
    assert isinstance(wrap_var(b'foo'), AnsibleUnsafeBytes)

def test_wrap_var_string():
    if False:
        for i in range(10):
            print('nop')
    assert isinstance(wrap_var('foo'), AnsibleUnsafeText)

def test_wrap_var_dict():
    if False:
        for i in range(10):
            print('nop')
    assert isinstance(wrap_var(dict(foo='bar')), dict)
    assert not isinstance(wrap_var(dict(foo='bar')), AnsibleUnsafe)
    assert isinstance(wrap_var(dict(foo=u'bar'))['foo'], AnsibleUnsafeText)

def test_wrap_var_dict_None():
    if False:
        for i in range(10):
            print('nop')
    assert wrap_var(dict(foo=None))['foo'] is None
    assert not isinstance(wrap_var(dict(foo=None))['foo'], AnsibleUnsafe)

def test_wrap_var_list():
    if False:
        return 10
    assert isinstance(wrap_var(['foo']), list)
    assert not isinstance(wrap_var(['foo']), AnsibleUnsafe)
    assert isinstance(wrap_var([u'foo'])[0], AnsibleUnsafeText)

def test_wrap_var_list_None():
    if False:
        i = 10
        return i + 15
    assert wrap_var([None])[0] is None
    assert not isinstance(wrap_var([None])[0], AnsibleUnsafe)

def test_wrap_var_set():
    if False:
        for i in range(10):
            print('nop')
    assert isinstance(wrap_var(set(['foo'])), set)
    assert not isinstance(wrap_var(set(['foo'])), AnsibleUnsafe)
    for item in wrap_var(set([u'foo'])):
        assert isinstance(item, AnsibleUnsafeText)

def test_wrap_var_set_None():
    if False:
        for i in range(10):
            print('nop')
    for item in wrap_var(set([None])):
        assert item is None
        assert not isinstance(item, AnsibleUnsafe)

def test_wrap_var_tuple():
    if False:
        for i in range(10):
            print('nop')
    assert isinstance(wrap_var(('foo',)), tuple)
    assert not isinstance(wrap_var(('foo',)), AnsibleUnsafe)
    assert isinstance(wrap_var(('foo',))[0], AnsibleUnsafe)

def test_wrap_var_tuple_None():
    if False:
        i = 10
        return i + 15
    assert wrap_var((None,))[0] is None
    assert not isinstance(wrap_var((None,))[0], AnsibleUnsafe)

def test_wrap_var_None():
    if False:
        return 10
    assert wrap_var(None) is None
    assert not isinstance(wrap_var(None), AnsibleUnsafe)

def test_wrap_var_unsafe_text():
    if False:
        for i in range(10):
            print('nop')
    assert isinstance(wrap_var(AnsibleUnsafeText(u'foo')), AnsibleUnsafeText)

def test_wrap_var_unsafe_bytes():
    if False:
        while True:
            i = 10
    assert isinstance(wrap_var(AnsibleUnsafeBytes(b'foo')), AnsibleUnsafeBytes)

def test_wrap_var_no_ref():
    if False:
        i = 10
        return i + 15
    thing = {'foo': {'bar': 'baz'}, 'bar': ['baz', 'qux'], 'baz': ('qux',), 'none': None, 'text': 'text'}
    wrapped_thing = wrap_var(thing)
    assert thing is not wrapped_thing
    assert thing['foo'] is not wrapped_thing['foo']
    assert thing['bar'][0] is not wrapped_thing['bar'][0]
    assert thing['baz'][0] is not wrapped_thing['baz'][0]
    assert thing['none'] is wrapped_thing['none']
    assert thing['text'] is not wrapped_thing['text']

def test_AnsibleUnsafeText():
    if False:
        while True:
            i = 10
    assert isinstance(AnsibleUnsafeText(u'foo'), AnsibleUnsafe)

def test_AnsibleUnsafeBytes():
    if False:
        i = 10
        return i + 15
    assert isinstance(AnsibleUnsafeBytes(b'foo'), AnsibleUnsafe)

def test_to_text_unsafe():
    if False:
        for i in range(10):
            print('nop')
    assert isinstance(to_text(AnsibleUnsafeBytes(b'foo')), AnsibleUnsafeText)
    assert to_text(AnsibleUnsafeBytes(b'foo')) == AnsibleUnsafeText(u'foo')

def test_to_bytes_unsafe():
    if False:
        i = 10
        return i + 15
    assert isinstance(to_bytes(AnsibleUnsafeText(u'foo')), AnsibleUnsafeBytes)
    assert to_bytes(AnsibleUnsafeText(u'foo')) == AnsibleUnsafeBytes(b'foo')