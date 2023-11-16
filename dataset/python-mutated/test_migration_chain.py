import pytest
from tribler.core.upgrade.tribler_db.decorator import migration
from tribler.core.upgrade.tribler_db.migration_chain import TriblerDatabaseMigrationChain
from tribler.core.utilities.path_util import Path
from tribler.core.utilities.pony_utils import db_session

def test_db_does_not_exist(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    ' Test that the migration chain does not execute if the database does not exist.'
    tribler_db_migration = TriblerDatabaseMigrationChain(state_dir=Path(tmpdir))
    assert not tribler_db_migration.execute()

@db_session
def test_db_execute(migration_chain: TriblerDatabaseMigrationChain):
    if False:
        for i in range(10):
            print('nop')
    ' Test that the migration chain executes all the migrations step by step.'
    migration_chain.db.version = 0

    @migration(execute_only_if_version=0)
    def migration1(*_, **__):
        if False:
            print('Hello World!')
        ...

    @migration(execute_only_if_version=1)
    def migration2(*_, **__):
        if False:
            return 10
        ...

    @migration(execute_only_if_version=99)
    def migration99(*_, **__):
        if False:
            while True:
                i = 10
        ...
    migration_chain.migrations = [migration1, migration2, migration99]
    assert migration_chain.execute()
    assert migration_chain.db.version == 2

@db_session
def test_db_execute_no_annotation(migration_chain: TriblerDatabaseMigrationChain):
    if False:
        i = 10
        return i + 15
    ' Test that the migration chain raises the NotImplementedError if the migration does not have the annotation.'

    def migration_without_annotation(*_, **__):
        if False:
            while True:
                i = 10
        ...
    migration_chain.migrations = [migration_without_annotation]
    with pytest.raises(NotImplementedError):
        migration_chain.execute()