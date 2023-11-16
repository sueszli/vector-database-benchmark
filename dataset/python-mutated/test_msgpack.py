from warehouse.i18n import LazyString
from warehouse.utils.msgpack import object_encode

def test_object_encode_passes_through():
    if False:
        while True:
            i = 10
    assert object_encode('foo') == 'foo'

def test_object_encode_converts_lazystring():
    if False:
        return 10

    def stringify(*args, **kwargs):
        if False:
            return 10
        return 'foo'
    ls = LazyString(stringify, 'foobar')
    assert object_encode(ls) == 'foo'