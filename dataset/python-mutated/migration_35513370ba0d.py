import random
from uuid import uuid4
import pytest
import sqlalchemy
from db import db
from journalist_app import create_app
from .helpers import bool_or_none, random_bool, random_chars, random_datetime

class UpgradeTester:

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        self.config = config
        self.app = create_app(config)

    def load_data(self):
        if False:
            print('Hello World!')
        with self.app.app_context():
            self.add_source()
            self.valid_source_id = 1
            db.session.commit()

    @staticmethod
    def add_source():
        if False:
            return 10
        filesystem_id = random_chars(96) if random_bool() else None
        params = {'uuid': str(uuid4()), 'filesystem_id': filesystem_id, 'journalist_designation': random_chars(50), 'flagged': bool_or_none(), 'last_updated': random_datetime(nullable=True), 'pending': bool_or_none(), 'interaction_count': random.randint(0, 1000)}
        sql = '\n        INSERT INTO sources (\n            uuid, filesystem_id, journalist_designation, flagged, last_updated,\n            pending, interaction_count\n        ) VALUES (\n            :uuid, :filesystem_id, :journalist_designation, :flagged, :last_updated,\n            :pending, :interaction_count\n        )\n        '
        db.engine.execute(sqlalchemy.text(sql), **params)

    def check_upgrade(self):
        if False:
            return 10
        "\n        Check the new `deleted_at` column\n\n        Querying `deleted_at` shouldn't cause an error, and no source\n        should already have it set.\n        "
        with self.app.app_context():
            sources = db.engine.execute(sqlalchemy.text('SELECT * FROM sources WHERE deleted_at IS NOT NULL')).fetchall()
            assert len(sources) == 0

class DowngradeTester:

    def __init__(self, config):
        if False:
            while True:
                i = 10
        self.config = config
        self.app = create_app(config)

    def load_data(self):
        if False:
            return 10
        pass

    def check_downgrade(self):
        if False:
            return 10
        '\n        After downgrade, using `deleted_at` in a query should raise an exception\n        '
        with self.app.app_context(), pytest.raises(sqlalchemy.exc.OperationalError):
            db.engine.execute(sqlalchemy.text('SELECT * FROM sources WHERE deleted_at IS NOT NULL')).fetchall()