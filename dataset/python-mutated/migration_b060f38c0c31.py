import random
import uuid
from typing import Any, Dict
import pytest
from db import db
from journalist_app import create_app
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from .helpers import bool_or_none, random_chars, random_datetime
random.seed('ᕕ( ᐛ )ᕗ')

def add_submission(source_id):
    if False:
        i = 10
        return i + 15
    params = {'uuid': str(uuid.uuid4()), 'source_id': source_id, 'filename': random_chars(50), 'size': random.randint(0, 1024 * 1024 * 500), 'downloaded': bool_or_none(), 'checksum': random_chars(255, chars='0123456789abcdef')}
    sql = '\n    INSERT INTO submissions (uuid, source_id, filename, size, downloaded, checksum)\n    VALUES (:uuid, :source_id, :filename, :size, :downloaded, :checksum)\n    '
    db.engine.execute(text(sql), **params)

class UpgradeTester:
    """
    Verify that the Source.flagged column no longer exists.
    """
    source_count = 10
    original_sources: Dict[str, Any] = {}
    source_submissions: Dict[str, Any] = {}

    def __init__(self, config):
        if False:
            while True:
                i = 10
        self.config = config
        self.app = create_app(config)

    def load_data(self):
        if False:
            i = 10
            return i + 15
        with self.app.app_context():
            for i in range(self.source_count):
                self.add_source()
            self.original_sources = {s.uuid: s for s in db.engine.execute(text('SELECT * FROM sources')).fetchall()}
            for s in self.original_sources.values():
                for i in range(random.randint(0, 3)):
                    add_submission(s.id)
                self.source_submissions[s.id] = db.engine.execute(text('SELECT * FROM submissions WHERE source_id = :source_id'), source_id=s.id).fetchall()

    def add_source(self):
        if False:
            for i in range(10):
                print('nop')
        params = {'uuid': str(uuid.uuid4()), 'filesystem_id': random_chars(96), 'journalist_designation': random_chars(50), 'flagged': bool_or_none(), 'last_updated': random_datetime(nullable=True), 'pending': bool_or_none(), 'interaction_count': random.randint(0, 1000)}
        sql = '\n        INSERT INTO sources (uuid, filesystem_id,\n        journalist_designation, flagged, last_updated, pending,\n        interaction_count)\n        VALUES (:uuid, :filesystem_id, :journalist_designation,\n        :flagged, :last_updated, :pending, :interaction_count)\n        '
        db.engine.execute(text(sql), **params)

    def check_upgrade(self):
        if False:
            return 10
        with self.app.app_context():
            with pytest.raises(OperationalError, match='.*sources has no column named flagged.*'):
                self.add_source()
            sources = db.engine.execute(text('SELECT * FROM sources')).fetchall()
            assert len(sources) == len(self.original_sources)
            for source in sources:
                assert not hasattr(source, 'flagged')
                original_source = self.original_sources[source.uuid]
                assert source.id == original_source.id
                assert source.journalist_designation == original_source.journalist_designation
                assert source.last_updated == original_source.last_updated
                assert source.pending == original_source.pending
                assert source.interaction_count == original_source.interaction_count
                source_submissions = db.engine.execute(text('SELECT * FROM submissions WHERE source_id = :source_id'), source_id=source.id).fetchall()
                assert source_submissions == self.source_submissions[source.id]

class DowngradeTester:
    """
    Verify that the Source.flagged column has been recreated properly.
    """
    source_count = 10
    original_sources: Dict[str, Any] = {}
    source_submissions: Dict[str, Any] = {}

    def __init__(self, config):
        if False:
            print('Hello World!')
        self.config = config
        self.app = create_app(config)

    def add_source(self):
        if False:
            while True:
                i = 10
        params = {'uuid': str(uuid.uuid4()), 'filesystem_id': random_chars(96), 'journalist_designation': random_chars(50), 'last_updated': random_datetime(nullable=True), 'pending': bool_or_none(), 'interaction_count': random.randint(0, 1000), 'deleted_at': None}
        sql = '\n        INSERT INTO sources (\n        uuid, filesystem_id, journalist_designation, last_updated, pending,\n        interaction_count\n        ) VALUES (\n        :uuid, :filesystem_id, :journalist_designation, :last_updated, :pending,\n        :interaction_count\n        )\n        '
        db.engine.execute(text(sql), **params)

    def load_data(self):
        if False:
            for i in range(10):
                print('nop')
        with self.app.app_context():
            for i in range(self.source_count):
                self.add_source()
            self.original_sources = {s.uuid: s for s in db.engine.execute(text('SELECT * FROM sources')).fetchall()}
            for s in self.original_sources.values():
                for i in range(random.randint(0, 3)):
                    add_submission(s.id)
                self.source_submissions[s.id] = db.engine.execute(text('SELECT * FROM submissions WHERE source_id = :source_id'), source_id=s.id).fetchall()

    def check_downgrade(self):
        if False:
            i = 10
            return i + 15
        with self.app.app_context():
            sources = db.engine.execute(text('SELECT * FROM sources')).fetchall()
            assert len(sources) == len(self.original_sources)
            for source in sources:
                assert hasattr(source, 'flagged')
                original_source = self.original_sources[source.uuid]
                assert source.id == original_source.id
                assert source.journalist_designation == original_source.journalist_designation
                assert source.last_updated == original_source.last_updated
                assert source.pending == original_source.pending
                assert source.interaction_count == original_source.interaction_count
                assert not hasattr(original_source, 'flagged')
                assert source.flagged is None
                source_submissions = db.engine.execute(text('SELECT * FROM submissions WHERE source_id = :source_id'), source_id=source.id).fetchall()
                assert source_submissions == self.source_submissions[source.id]