import random
import uuid
from db import db
from journalist_app import create_app
from sqlalchemy import text
from sqlalchemy.exc import NoSuchColumnError
from .helpers import bool_or_none, random_bool, random_chars, random_datetime
random.seed('ᕕ( ᐛ )ᕗ')

def add_source():
    if False:
        i = 10
        return i + 15
    filesystem_id = random_chars(96) if random_bool() else None
    params = {'filesystem_id': filesystem_id, 'uuid': str(uuid.uuid4()), 'journalist_designation': random_chars(50), 'flagged': bool_or_none(), 'last_updated': random_datetime(nullable=True), 'pending': bool_or_none(), 'interaction_count': random.randint(0, 1000)}
    sql = 'INSERT INTO sources (filesystem_id, uuid,\n                journalist_designation, flagged, last_updated, pending,\n                interaction_count)\n             VALUES (:filesystem_id, :uuid, :journalist_designation,\n                :flagged, :last_updated, :pending, :interaction_count)\n          '
    db.engine.execute(text(sql), **params)

class UpgradeTester:
    """This migration verifies that the UUID column now exists, and that
    the data migration completed successfully.
    """
    SOURCE_NUM = 200

    def __init__(self, config):
        if False:
            return 10
        self.config = config
        self.app = create_app(config)

    def load_data(self):
        if False:
            print('Hello World!')
        with self.app.app_context():
            for _ in range(self.SOURCE_NUM):
                add_source()
            for sid in range(1, self.SOURCE_NUM, 8):
                for _ in range(random.randint(1, 3)):
                    self.add_submission(sid)
            for sid in range(self.SOURCE_NUM, self.SOURCE_NUM + 50):
                self.add_submission(sid)
            db.session.commit()

    @staticmethod
    def add_submission(source_id):
        if False:
            while True:
                i = 10
        params = {'source_id': source_id, 'filename': random_chars(50), 'size': random.randint(0, 1024 * 1024 * 500), 'downloaded': bool_or_none()}
        sql = 'INSERT INTO submissions (source_id, filename, size,\n                    downloaded)\n                 VALUES (:source_id, :filename, :size, :downloaded)\n              '
        db.engine.execute(text(sql), **params)

    def check_upgrade(self):
        if False:
            for i in range(10):
                print('nop')
        with self.app.app_context():
            submissions = db.engine.execute(text('SELECT * FROM submissions')).fetchall()
            for submission in submissions:
                assert submission.uuid is not None

class DowngradeTester:
    SOURCE_NUM = 200

    def __init__(self, config):
        if False:
            while True:
                i = 10
        self.config = config
        self.app = create_app(config)

    def load_data(self):
        if False:
            print('Hello World!')
        with self.app.app_context():
            for _ in range(self.SOURCE_NUM):
                add_source()
            for sid in range(1, self.SOURCE_NUM, 8):
                for _ in range(random.randint(1, 3)):
                    self.add_submission(sid)
            for sid in range(self.SOURCE_NUM, self.SOURCE_NUM + 50):
                self.add_submission(sid)
            db.session.commit()

    @staticmethod
    def add_submission(source_id):
        if False:
            while True:
                i = 10
        params = {'source_id': source_id, 'uuid': str(uuid.uuid4()), 'filename': random_chars(50), 'size': random.randint(0, 1024 * 1024 * 500), 'downloaded': bool_or_none()}
        sql = 'INSERT INTO submissions (source_id, uuid, filename, size,\n                    downloaded)\n                 VALUES (:source_id, :uuid, :filename, :size, :downloaded)\n              '
        db.engine.execute(text(sql), **params)

    def check_downgrade(self):
        if False:
            while True:
                i = 10
        'Verify that the UUID column is now gone, but otherwise the table\n        has the expected number of rows.\n        '
        with self.app.app_context():
            sql = 'SELECT * FROM submissions'
            submissions = db.engine.execute(text(sql)).fetchall()
            for submission in submissions:
                try:
                    assert submission['uuid'] is None
                except NoSuchColumnError:
                    pass