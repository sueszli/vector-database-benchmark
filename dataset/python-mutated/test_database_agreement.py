import os
import shutil
import tempfile
from hypothesis import strategies as st
from hypothesis.database import DirectoryBasedExampleDatabase, InMemoryExampleDatabase
from hypothesis.stateful import Bundle, RuleBasedStateMachine, rule

class DatabaseComparison(RuleBasedStateMachine):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.tempd = tempfile.mkdtemp()
        exampledir = os.path.join(self.tempd, 'examples')
        self.dbs = [DirectoryBasedExampleDatabase(exampledir), InMemoryExampleDatabase(), DirectoryBasedExampleDatabase(exampledir)]
    keys = Bundle('keys')
    values = Bundle('values')

    @rule(target=keys, k=st.binary())
    def k(self, k):
        if False:
            print('Hello World!')
        return k

    @rule(target=values, v=st.binary())
    def v(self, v):
        if False:
            return 10
        return v

    @rule(k=keys, v=values)
    def save(self, k, v):
        if False:
            return 10
        for db in self.dbs:
            db.save(k, v)

    @rule(k=keys, v=values)
    def delete(self, k, v):
        if False:
            while True:
                i = 10
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
            for i in range(10):
                print('nop')
        last = None
        last_db = None
        for db in self.dbs:
            keys = set(db.fetch(k))
            if last is not None:
                assert last == keys, (last_db, db)
            last = keys
            last_db = db

    def teardown(self):
        if False:
            for i in range(10):
                print('nop')
        shutil.rmtree(self.tempd)
TestDBs = DatabaseComparison.TestCase