from datetime import datetime
from datetime import timezone
from uuid import uuid4
import pytest
from markupsafe import Markup
from flask.json.tag import JSONTag
from flask.json.tag import TaggedJSONSerializer

@pytest.mark.parametrize('data', ({' t': (1, 2, 3)}, {' t__': b'a'}, {' di': ' di'}, {'x': (1, 2, 3), 'y': 4}, (1, 2, 3), [(1, 2, 3)], b'\xff', Markup('<html>'), uuid4(), datetime.now(tz=timezone.utc).replace(microsecond=0)))
def test_dump_load_unchanged(data):
    if False:
        for i in range(10):
            print('nop')
    s = TaggedJSONSerializer()
    assert s.loads(s.dumps(data)) == data

def test_duplicate_tag():
    if False:
        print('Hello World!')

    class TagDict(JSONTag):
        key = ' d'
    s = TaggedJSONSerializer()
    pytest.raises(KeyError, s.register, TagDict)
    s.register(TagDict, force=True, index=0)
    assert isinstance(s.tags[' d'], TagDict)
    assert isinstance(s.order[0], TagDict)

def test_custom_tag():
    if False:
        while True:
            i = 10

    class Foo:

        def __init__(self, data):
            if False:
                print('Hello World!')
            self.data = data

    class TagFoo(JSONTag):
        __slots__ = ()
        key = ' f'

        def check(self, value):
            if False:
                i = 10
                return i + 15
            return isinstance(value, Foo)

        def to_json(self, value):
            if False:
                return 10
            return self.serializer.tag(value.data)

        def to_python(self, value):
            if False:
                i = 10
                return i + 15
            return Foo(value)
    s = TaggedJSONSerializer()
    s.register(TagFoo)
    assert s.loads(s.dumps(Foo('bar'))).data == 'bar'

def test_tag_interface():
    if False:
        while True:
            i = 10
    t = JSONTag(None)
    pytest.raises(NotImplementedError, t.check, None)
    pytest.raises(NotImplementedError, t.to_json, None)
    pytest.raises(NotImplementedError, t.to_python, None)

def test_tag_order():
    if False:
        for i in range(10):
            print('nop')

    class Tag1(JSONTag):
        key = ' 1'

    class Tag2(JSONTag):
        key = ' 2'
    s = TaggedJSONSerializer()
    s.register(Tag1, index=-1)
    assert isinstance(s.order[-2], Tag1)
    s.register(Tag2, index=None)
    assert isinstance(s.order[-1], Tag2)