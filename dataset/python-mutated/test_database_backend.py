import os
import re
import tempfile
import zipfile
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from shutil import make_archive, rmtree
from typing import Iterator, Optional, Tuple
import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.database import DirectoryBasedExampleDatabase, ExampleDatabase, GitHubArtifactDatabase, InMemoryExampleDatabase, MultiplexedDatabase, ReadOnlyDatabase
from hypothesis.errors import HypothesisWarning
from hypothesis.stateful import Bundle, RuleBasedStateMachine, rule
from hypothesis.strategies import binary, lists, tuples
small_settings = settings(max_examples=50)

@given(lists(tuples(binary(), binary())))
@small_settings
def test_backend_returns_what_you_put_in(xs):
    if False:
        i = 10
        return i + 15
    backend = InMemoryExampleDatabase()
    mapping = {}
    for (key, value) in xs:
        mapping.setdefault(key, set()).add(value)
        backend.save(key, value)
    for (key, values) in mapping.items():
        backend_contents = list(backend.fetch(key))
        distinct_backend_contents = set(backend_contents)
        assert len(backend_contents) == len(distinct_backend_contents)
        assert distinct_backend_contents == set(values)

def test_can_delete_keys():
    if False:
        print('Hello World!')
    backend = InMemoryExampleDatabase()
    backend.save(b'foo', b'bar')
    backend.save(b'foo', b'baz')
    backend.delete(b'foo', b'bar')
    assert list(backend.fetch(b'foo')) == [b'baz']

def test_default_database_is_in_memory():
    if False:
        print('Hello World!')
    assert isinstance(ExampleDatabase(), InMemoryExampleDatabase)

def test_default_on_disk_database_is_dir(tmpdir):
    if False:
        i = 10
        return i + 15
    assert isinstance(ExampleDatabase(tmpdir.join('foo')), DirectoryBasedExampleDatabase)

def test_selects_directory_based_if_already_directory(tmpdir):
    if False:
        i = 10
        return i + 15
    path = str(tmpdir.join('hi.sqlite3'))
    DirectoryBasedExampleDatabase(path).save(b'foo', b'bar')
    assert isinstance(ExampleDatabase(path), DirectoryBasedExampleDatabase)

def test_does_not_error_when_fetching_when_not_exist(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    db = DirectoryBasedExampleDatabase(tmpdir.join('examples'))
    db.fetch(b'foo')

@pytest.fixture(scope='function', params=['memory', 'directory'])
def exampledatabase(request, tmpdir):
    if False:
        while True:
            i = 10
    if request.param == 'memory':
        return ExampleDatabase()
    assert request.param == 'directory'
    return DirectoryBasedExampleDatabase(str(tmpdir.join('examples')))

def test_can_delete_a_key_that_is_not_present(exampledatabase):
    if False:
        while True:
            i = 10
    exampledatabase.delete(b'foo', b'bar')

def test_can_fetch_a_key_that_is_not_present(exampledatabase):
    if False:
        print('Hello World!')
    assert list(exampledatabase.fetch(b'foo')) == []

def test_saving_a_key_twice_fetches_it_once(exampledatabase):
    if False:
        for i in range(10):
            print('nop')
    exampledatabase.save(b'foo', b'bar')
    exampledatabase.save(b'foo', b'bar')
    assert list(exampledatabase.fetch(b'foo')) == [b'bar']

def test_can_close_a_database_after_saving(exampledatabase):
    if False:
        i = 10
        return i + 15
    exampledatabase.save(b'foo', b'bar')

def test_class_name_is_in_repr(exampledatabase):
    if False:
        return 10
    assert type(exampledatabase).__name__ in repr(exampledatabase)

def test_an_absent_value_is_present_after_it_moves(exampledatabase):
    if False:
        return 10
    exampledatabase.move(b'a', b'b', b'c')
    assert next(exampledatabase.fetch(b'b')) == b'c'

def test_an_absent_value_is_present_after_it_moves_to_self(exampledatabase):
    if False:
        i = 10
        return i + 15
    exampledatabase.move(b'a', b'a', b'b')
    assert next(exampledatabase.fetch(b'a')) == b'b'

def test_two_directory_databases_can_interact(tmpdir):
    if False:
        while True:
            i = 10
    path = str(tmpdir)
    db1 = DirectoryBasedExampleDatabase(path)
    db2 = DirectoryBasedExampleDatabase(path)
    db1.save(b'foo', b'bar')
    assert list(db2.fetch(b'foo')) == [b'bar']
    db2.save(b'foo', b'bar')
    db2.save(b'foo', b'baz')
    assert sorted(db1.fetch(b'foo')) == [b'bar', b'baz']

def test_can_handle_disappearing_files(tmpdir, monkeypatch):
    if False:
        print('Hello World!')
    path = str(tmpdir)
    db = DirectoryBasedExampleDatabase(path)
    db.save(b'foo', b'bar')
    base_listdir = os.listdir
    monkeypatch.setattr(os, 'listdir', lambda d: [*base_listdir(d), 'this-does-not-exist'])
    assert list(db.fetch(b'foo')) == [b'bar']

def test_readonly_db_is_not_writable():
    if False:
        while True:
            i = 10
    inner = InMemoryExampleDatabase()
    wrapped = ReadOnlyDatabase(inner)
    inner.save(b'key', b'value')
    inner.save(b'key', b'value2')
    wrapped.delete(b'key', b'value')
    wrapped.move(b'key', b'key2', b'value2')
    wrapped.save(b'key', b'value3')
    assert set(wrapped.fetch(b'key')) == {b'value', b'value2'}
    assert set(wrapped.fetch(b'key2')) == set()

def test_multiplexed_dbs_read_and_write_all():
    if False:
        while True:
            i = 10
    a = InMemoryExampleDatabase()
    b = InMemoryExampleDatabase()
    multi = MultiplexedDatabase(a, b)
    a.save(b'a', b'aa')
    b.save(b'b', b'bb')
    multi.save(b'c', b'cc')
    multi.move(b'a', b'b', b'aa')
    for db in (a, b, multi):
        assert set(db.fetch(b'a')) == set()
        assert set(db.fetch(b'c')) == {b'cc'}
    got = list(multi.fetch(b'b'))
    assert len(got) == 2
    assert set(got) == {b'aa', b'bb'}
    multi.delete(b'c', b'cc')
    for db in (a, b, multi):
        assert set(db.fetch(b'c')) == set()

def test_ga_require_readonly_wrapping():
    if False:
        for i in range(10):
            print('nop')
    'Test that GitHubArtifactDatabase requires wrapping around ReadOnlyDatabase'
    database = GitHubArtifactDatabase('test', 'test')
    with pytest.raises(RuntimeError, match=re.escape(database._read_only_message)):
        database.save(b'foo', b'bar')
    with pytest.raises(RuntimeError):
        database.move(b'foo', b'bar', b'foobar')
    with pytest.raises(RuntimeError):
        database.delete(b'foo', b'bar')
    database = ReadOnlyDatabase(database)
    database.save(b'foo', b'bar')
    database.move(b'foo', b'bar', b'foobar')
    database.delete(b'foo', b'bar')

@contextmanager
def ga_empty_artifact(date: Optional[datetime]=None, path: Optional[Path]=None) -> Iterator[Tuple[Path, Path]]:
    if False:
        while True:
            i = 10
    'Creates an empty GitHub artifact.'
    if date:
        timestamp = date.isoformat().replace(':', '_')
    else:
        timestamp = datetime.now(timezone.utc).isoformat().replace(':', '_')
    temp_dir = None
    if not path:
        temp_dir = tempfile.mkdtemp()
        path = Path(temp_dir) / 'github-artifacts'
    path.mkdir(parents=True, exist_ok=True)
    zip_path = path / f'{timestamp}.zip'
    with zipfile.ZipFile(zip_path, 'w'):
        pass
    try:
        yield (path, zip_path)
    finally:
        if temp_dir:
            rmtree(temp_dir)

def test_ga_empty_read():
    if False:
        i = 10
        return i + 15
    'Tests that an inexistent key returns nothing.'
    with ga_empty_artifact() as (path, _):
        database = GitHubArtifactDatabase('test', 'test', path=path)
        assert list(database.fetch(b'foo')) == []

def test_ga_initialize():
    if False:
        while True:
            i = 10
    "\n    Tests that the database is initialized when a new artifact is found.\n    As well that initialization doesn't happen again on the next fetch.\n    "
    now = datetime.now(timezone.utc)
    with ga_empty_artifact(date=now - timedelta(hours=2)) as (path, _):
        database = GitHubArtifactDatabase('test', 'test', path=path)
        list(database.fetch(b''))
        initial_artifact = database._artifact
        assert initial_artifact
        assert database._artifact
        assert database._access_cache is not None
        with ga_empty_artifact(date=now, path=path) as (path, _):
            list(database.fetch(b''))
            assert database._initialized
            assert database._artifact == initial_artifact

def test_ga_no_artifact(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    'Tests that the database is disabled when no artifact is found.'
    database = GitHubArtifactDatabase('test', 'test', path=tmp_path)
    with pytest.warns(HypothesisWarning):
        assert list(database.fetch(b'')) == []
    assert database._disabled is True
    assert list(database.fetch(b'')) == []

def test_ga_corrupted_artifact():
    if False:
        return 10
    'Tests that corrupted artifacts are properly detected and warned about.'
    with ga_empty_artifact() as (path, zip_path):
        with open(zip_path, 'rb+') as f:
            f.write(b'\x00\x01\x00\x01')
        database = GitHubArtifactDatabase('test', 'test', path=path)
        with pytest.warns(HypothesisWarning):
            assert list(database.fetch(b'')) == []
        assert database._disabled is True

def test_ga_deletes_old_artifacts():
    if False:
        i = 10
        return i + 15
    'Tests that old artifacts are automatically deleted.'
    now = datetime.now(timezone.utc)
    with ga_empty_artifact(date=now) as (path, file_now):
        with ga_empty_artifact(date=now - timedelta(hours=2), path=path) as (_, file_old):
            database = GitHubArtifactDatabase('test', 'test', path=path)
            list(database.fetch(b''))
            assert file_now.exists()
            assert not file_old.exists()

def test_ga_triggers_fetching(monkeypatch, tmp_path):
    if False:
        while True:
            i = 10
    'Tests whether an artifact fetch is triggered, and an expired artifact is deleted.'
    with ga_empty_artifact() as (_, artifact):

        def fake_fetch_artifact(self) -> Optional[Path]:
            if False:
                while True:
                    i = 10
            return artifact
        monkeypatch.setattr(GitHubArtifactDatabase, '_fetch_artifact', fake_fetch_artifact)
        database = GitHubArtifactDatabase('test', 'test', path=tmp_path, cache_timeout=timedelta(days=1))
        list(database.fetch(b''))
        assert not database._disabled
        assert database._initialized
        assert database._artifact == artifact
        now = datetime.now(timezone.utc)
        with ga_empty_artifact(date=now - timedelta(days=2)) as (path_with_artifact, old_artifact):
            database = GitHubArtifactDatabase('test', 'test', path=path_with_artifact, cache_timeout=timedelta(days=1))
            list(database.fetch(b''))
            assert not database._disabled
            assert database._initialized
            assert database._artifact == artifact
            assert not old_artifact.exists()

def test_ga_fallback_expired(monkeypatch):
    if False:
        while True:
            i = 10
    '\n    Tests that the fallback to an expired artifact is triggered\n    if fetching a new one fails. This allows for (by example) offline development.\n    '
    now = datetime.now(timezone.utc)
    with ga_empty_artifact(date=now - timedelta(days=2)) as (path, artifact):
        database = GitHubArtifactDatabase('test', 'test', path=path, cache_timeout=timedelta(days=1))

        def fake_fetch_artifact(self) -> Optional[Path]:
            if False:
                while True:
                    i = 10
            return None
        monkeypatch.setattr(GitHubArtifactDatabase, '_fetch_artifact', fake_fetch_artifact)
        with pytest.warns(HypothesisWarning):
            list(database.fetch(b''))
        assert not database._disabled
        assert database._initialized
        assert database._artifact == artifact

class GitHubArtifactMocks(RuleBasedStateMachine):
    """
    This is a state machine that tests agreement of GitHubArtifactDatabase
    with DirectoryBasedExampleDatabase (as a reference implementation).
    """

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.temp_directory = Path(tempfile.mkdtemp())
        self.path = self.temp_directory / 'github-artifacts'
        timestamp = datetime.now(timezone.utc).isoformat().replace(':', '_')
        self.zip_destination = self.path / f'{timestamp}.zip'
        self.zip_content_path = self.path / timestamp
        self.zip_content_path.mkdir(parents=True, exist_ok=True)
        self.directory_db = DirectoryBasedExampleDatabase(str(self.zip_content_path))
        self.zip_db = GitHubArtifactDatabase('mock', 'mock', path=self.path)
        self._archive_directory_db()
        self.zip_db._initialize_db()

    def _make_zip(self, tree_path: Path, zip_path: Path):
        if False:
            for i in range(10):
                print('nop')
        destination = zip_path.parent.absolute() / zip_path.stem
        make_archive(str(destination), 'zip', root_dir=tree_path)

    def _archive_directory_db(self):
        if False:
            return 10
        for file in self.path.glob('*.zip'):
            file.unlink()
        self._make_zip(self.zip_content_path, self.zip_destination)
    keys = Bundle('keys')
    values = Bundle('values')

    @rule(target=keys, k=st.binary())
    def k(self, k):
        if False:
            while True:
                i = 10
        return k

    @rule(target=values, v=st.binary())
    def v(self, v):
        if False:
            return 10
        return v

    @rule(k=keys, v=values)
    def save(self, k, v):
        if False:
            for i in range(10):
                print('nop')
        self.directory_db.save(k, v)
        self._archive_directory_db()
        self.zip_db = GitHubArtifactDatabase('mock', 'mock', path=self.path)
        self.zip_db._initialize_db()

    @rule(k=keys)
    def values_agree(self, k):
        if False:
            i = 10
            return i + 15
        v1 = set(self.directory_db.fetch(k))
        v2 = set(self.zip_db.fetch(k))
        assert v1 == v2
TestGADReads = GitHubArtifactMocks.TestCase

def test_gadb_coverage():
    if False:
        print('Hello World!')
    state = GitHubArtifactMocks()
    state.save(b'key', b'value')
    state.values_agree(b'key')