import os
import random
from uuid import uuid4
from db import db
from journalist_app import create_app
from sqlalchemy import text
from .helpers import bool_or_none, random_ascii_chars, random_bool, random_chars, random_datetime
TEST_DATA_DIR = '/tmp/securedrop/store'

def create_file_in_dummy_source_dir(filename):
    if False:
        return 10
    filesystem_id = 'dummy'
    basedir = os.path.join(TEST_DATA_DIR, filesystem_id)
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    path_to_file = os.path.join(basedir, filename)
    with open(path_to_file, 'a'):
        os.utime(path_to_file, None)

class UpgradeTester:
    """This migration verifies that any orphaned submission or reply data from
    deleted sources is also deleted.
    """

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        self.config = config
        self.app = create_app(config)
        self.journalist_id = None

    def load_data(self):
        if False:
            i = 10
            return i + 15
        with self.app.app_context():
            self.create_journalist()
            self.add_source()
            self.valid_source_id = 1
            deleted_source_id = 2
            self.add_submission(self.valid_source_id)
            self.add_submission(deleted_source_id)
            self.add_submission(deleted_source_id, with_file=False)
            self.add_submission(None)
            self.add_reply(self.journalist_id, self.valid_source_id)
            self.add_reply(self.journalist_id, deleted_source_id)
            self.add_reply(self.journalist_id, deleted_source_id, with_file=False)
            self.add_reply(self.journalist_id, None)
            db.session.commit()

    def create_journalist(self):
        if False:
            i = 10
            return i + 15
        if self.journalist_id is not None:
            raise RuntimeError('Journalist already created')
        params = {'uuid': str(uuid4()), 'username': random_chars(50), 'session_nonce': 0}
        sql = 'INSERT INTO journalists (uuid, username, session_nonce)\n                 VALUES (:uuid, :username, :session_nonce)\n              '
        self.journalist_id = db.engine.execute(text(sql), **params).lastrowid

    def add_reply(self, journalist_id, source_id, with_file=True):
        if False:
            print('Hello World!')
        filename = '1-' + random_ascii_chars(5) + '-' + random_ascii_chars(5) + '-reply.gpg'
        params = {'uuid': str(uuid4()), 'journalist_id': journalist_id, 'source_id': source_id, 'filename': filename, 'size': random.randint(0, 1024 * 1024 * 500), 'deleted_by_source': False}
        sql = 'INSERT INTO replies (journalist_id, uuid, source_id, filename,\n                    size, deleted_by_source)\n                 VALUES (:journalist_id, :uuid, :source_id, :filename, :size,\n                         :deleted_by_source)\n              '
        db.engine.execute(text(sql), **params)
        if with_file:
            create_file_in_dummy_source_dir(filename)

    @staticmethod
    def add_source():
        if False:
            i = 10
            return i + 15
        filesystem_id = random_chars(96) if random_bool() else None
        params = {'uuid': str(uuid4()), 'filesystem_id': filesystem_id, 'journalist_designation': random_chars(50), 'flagged': bool_or_none(), 'last_updated': random_datetime(nullable=True), 'pending': bool_or_none(), 'interaction_count': random.randint(0, 1000)}
        sql = 'INSERT INTO sources (uuid, filesystem_id,\n                    journalist_designation, flagged, last_updated, pending,\n                    interaction_count)\n                 VALUES (:uuid, :filesystem_id, :journalist_designation,\n                    :flagged, :last_updated, :pending, :interaction_count)\n              '
        db.engine.execute(text(sql), **params)

    def add_submission(self, source_id, with_file=True):
        if False:
            print('Hello World!')
        filename = '1-' + random_ascii_chars(5) + '-' + random_ascii_chars(5) + '-doc.gz.gpg'
        params = {'uuid': str(uuid4()), 'source_id': source_id, 'filename': filename, 'size': random.randint(0, 1024 * 1024 * 500), 'downloaded': bool_or_none()}
        sql = 'INSERT INTO submissions (uuid, source_id, filename, size,\n                    downloaded)\n                 VALUES (:uuid, :source_id, :filename, :size, :downloaded)\n              '
        db.engine.execute(text(sql), **params)
        if with_file:
            create_file_in_dummy_source_dir(filename)

    def check_upgrade(self):
        if False:
            for i in range(10):
                print('nop')
        with self.app.app_context():
            submissions = db.engine.execute(text('SELECT * FROM submissions')).fetchall()
            assert len(submissions) == 1
            for submission in submissions:
                assert submission.source_id == self.valid_source_id
            replies = db.engine.execute(text('SELECT * FROM replies')).fetchall()
            assert len(replies) == 1
            for reply in replies:
                assert reply.source_id == self.valid_source_id

class DowngradeTester:

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        self.config = config

    def load_data(self):
        if False:
            return 10
        pass

    def check_downgrade(self):
        if False:
            for i in range(10):
                print('nop')
        pass