import pytest
from fakeredis import FakeRedis
from hypothesis import strategies as st
from hypothesis.database import InMemoryExampleDatabase
from hypothesis.errors import InvalidArgument
from hypothesis.extra.redis import RedisExampleDatabase
from hypothesis.stateful import Bundle, RuleBasedStateMachine, rule

@pytest.mark.parametrize('kw', [{'redis': 'not a redis instance'}, {'redis': FakeRedis(), 'expire_after': 10}, {'redis': FakeRedis(), 'key_prefix': 'not a bytestring'}])
def test_invalid_args_raise(kw):
    if False:
        while True:
            i = 10
    with pytest.raises(InvalidArgument):
        RedisExampleDatabase(**kw)

def test_all_methods():
    if False:
        print('Hello World!')
    db = RedisExampleDatabase(FakeRedis())
    db.save(b'key1', b'value')
    assert list(db.fetch(b'key1')) == [b'value']
    db.move(b'key1', b'key2', b'value')
    assert list(db.fetch(b'key1')) == []
    assert list(db.fetch(b'key2')) == [b'value']
    db.delete(b'key2', b'value')
    assert list(db.fetch(b'key2')) == []
    db.delete(b'key2', b'unknown value')

class DatabaseComparison(RuleBasedStateMachine):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.dbs = [InMemoryExampleDatabase(), RedisExampleDatabase(FakeRedis())]
    keys = Bundle('keys')
    values = Bundle('values')

    @rule(target=keys, k=st.binary())
    def k(self, k):
        if False:
            i = 10
            return i + 15
        return k

    @rule(target=values, v=st.binary())
    def v(self, v):
        if False:
            print('Hello World!')
        return v

    @rule(k=keys, v=values)
    def save(self, k, v):
        if False:
            print('Hello World!')
        for db in self.dbs:
            db.save(k, v)

    @rule(k=keys, v=values)
    def delete(self, k, v):
        if False:
            for i in range(10):
                print('nop')
        for db in self.dbs:
            db.delete(k, v)

    @rule(k1=keys, k2=keys, v=values)
    def move(self, k1, k2, v):
        if False:
            print('Hello World!')
        for db in self.dbs:
            db.move(k1, k2, v)

    @rule(k=keys)
    def values_agree(self, k):
        if False:
            return 10
        last = None
        last_db = None
        for db in self.dbs:
            keys = set(db.fetch(k))
            if last is not None:
                assert last == keys, (last_db, db)
            last = keys
            last_db = db
TestDBs = DatabaseComparison.TestCase