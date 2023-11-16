from unittest.mock import Mock, patch
import pytest
from tribler.core.upgrade.knowledge_to_triblerdb.migration import MigrationKnowledgeToTriblerDB
from tribler.core.upgrade.tags_to_knowledge.previous_dbs.knowledge_db import KnowledgeDatabase
from tribler.core.utilities.path_util import Path
from tribler.core.utilities.simpledefs import STATEDIR_DB_DIR

@pytest.fixture
def migration(tmp_path: Path):
    if False:
        i = 10
        return i + 15
    db_dir = tmp_path / STATEDIR_DB_DIR
    db_dir.mkdir()
    migration = MigrationKnowledgeToTriblerDB(tmp_path)
    return migration

def test_no_knowledge_db(migration: MigrationKnowledgeToTriblerDB):
    if False:
        for i in range(10):
            print('nop')
    assert not migration.run()
    assert not migration.knowledge_db_path.exists()
    assert not migration.tribler_db_path.exists()

def test_move_file(migration: MigrationKnowledgeToTriblerDB):
    if False:
        while True:
            i = 10
    KnowledgeDatabase(str(migration.knowledge_db_path)).shutdown()
    assert migration.knowledge_db_path.exists()
    assert not migration.tribler_db_path.exists()
    assert migration.run()
    assert not migration.knowledge_db_path.exists()
    assert migration.tribler_db_path.exists()

@patch('tribler.core.upgrade.knowledge_to_triblerdb.migration.shutil.move', Mock(side_effect=FileNotFoundError))
def test_exception(migration: MigrationKnowledgeToTriblerDB):
    if False:
        i = 10
        return i + 15
    KnowledgeDatabase(str(migration.knowledge_db_path)).shutdown()
    assert migration.knowledge_db_path.exists()
    assert not migration.tribler_db_path.exists()
    assert not migration.run()
    assert migration.knowledge_db_path.exists()
    assert not migration.tribler_db_path.exists()