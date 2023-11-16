from hypothesis import assume, core, find, given, settings, strategies as st
from hypothesis.database import ExampleDatabase, GitHubArtifactDatabase, InMemoryExampleDatabase, ReadOnlyDatabase
from hypothesis.errors import NoSuchExample, Unsatisfiable
from hypothesis.internal.entropy import deterministic_PRNG
from tests.common.utils import all_values, non_covering_examples

def has_a_non_zero_byte(x):
    if False:
        while True:
            i = 10
    return any(bytes(x))

def test_saves_incremental_steps_in_database():
    if False:
        for i in range(10):
            print('nop')
    key = b'a database key'
    database = InMemoryExampleDatabase()
    find(st.binary(min_size=10), has_a_non_zero_byte, settings=settings(database=database), database_key=key)
    assert len(all_values(database)) > 1

def test_clears_out_database_as_things_get_boring():
    if False:
        for i in range(10):
            print('nop')
    key = b'a database key'
    database = InMemoryExampleDatabase()
    do_we_care = True

    def stuff():
        if False:
            return 10
        try:
            find(st.binary(min_size=50), lambda x: do_we_care and has_a_non_zero_byte(x), settings=settings(database=database, max_examples=10), database_key=key)
        except NoSuchExample:
            pass
    stuff()
    assert len(non_covering_examples(database)) > 1
    do_we_care = False
    stuff()
    initial = len(non_covering_examples(database))
    assert initial > 0
    for _ in range(initial):
        stuff()
        keys = len(non_covering_examples(database))
        if not keys:
            break
    else:
        raise AssertionError

def test_trashes_invalid_examples():
    if False:
        print('Hello World!')
    key = b'a database key'
    database = InMemoryExampleDatabase()
    invalid = set()

    def stuff():
        if False:
            i = 10
            return i + 15
        try:

            def condition(x):
                if False:
                    print('Hello World!')
                assume(x not in invalid)
                return not invalid and has_a_non_zero_byte(x)
            return find(st.binary(min_size=5), condition, settings=settings(database=database), database_key=key)
        except (Unsatisfiable, NoSuchExample):
            pass
    with deterministic_PRNG():
        value = stuff()
    original = len(all_values(database))
    assert original > 1
    invalid.add(value)
    with deterministic_PRNG():
        stuff()
    assert len(all_values(database)) < original

def test_respects_max_examples_in_database_usage():
    if False:
        while True:
            i = 10
    key = b'a database key'
    database = InMemoryExampleDatabase()
    do_we_care = True
    counter = [0]

    def check(x):
        if False:
            print('Hello World!')
        counter[0] += 1
        return do_we_care and has_a_non_zero_byte(x)

    def stuff():
        if False:
            while True:
                i = 10
        try:
            find(st.binary(min_size=100), check, settings=settings(database=database, max_examples=10), database_key=key)
        except NoSuchExample:
            pass
    with deterministic_PRNG():
        stuff()
    assert len(all_values(database)) > 10
    do_we_care = False
    counter[0] = 0
    stuff()
    assert counter == [10]

def test_does_not_use_database_when_seed_is_forced(monkeypatch):
    if False:
        i = 10
        return i + 15
    monkeypatch.setattr(core, 'global_force_seed', 42)
    database = InMemoryExampleDatabase()
    database.fetch = None

    @settings(database=database)
    @given(st.integers())
    def test(i):
        if False:
            while True:
                i = 10
        pass
    test()

@given(st.binary(), st.binary())
def test_database_not_created_when_not_used(tmp_path_factory, key, value):
    if False:
        for i in range(10):
            print('nop')
    path = tmp_path_factory.mktemp('hypothesis') / 'examples'
    assert not path.exists()
    database = ExampleDatabase(path)
    assert not list(database.fetch(key))
    assert not path.exists()
    database.save(key, value)
    assert path.exists()
    assert list(database.fetch(key)) == [value]

def test_ga_database_not_created_when_not_used(tmp_path_factory):
    if False:
        for i in range(10):
            print('nop')
    path = tmp_path_factory.mktemp('hypothesis') / 'github-actions'
    assert not path.exists()
    ReadOnlyDatabase(GitHubArtifactDatabase('mock', 'mock', path=path))
    assert not path.exists()