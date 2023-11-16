import random
import string
import uuid
from db import db
from journalist_app import create_app
from sqlalchemy import text
from sqlalchemy.exc import NoSuchColumnError
from .helpers import bool_or_none, random_bool, random_bytes, random_chars, random_datetime, random_username
random.seed('ᕕ( ᐛ )ᕗ')

def add_source():
    if False:
        print('Hello World!')
    filesystem_id = random_chars(96) if random_bool() else None
    params = {'uuid': str(uuid.uuid4()), 'filesystem_id': filesystem_id, 'journalist_designation': random_chars(50), 'flagged': bool_or_none(), 'last_updated': random_datetime(nullable=True), 'pending': bool_or_none(), 'interaction_count': random.randint(0, 1000)}
    sql = 'INSERT INTO sources (uuid, filesystem_id,\n                journalist_designation, flagged, last_updated, pending,\n                interaction_count)\n             VALUES (:uuid, :filesystem_id, :journalist_designation,\n                :flagged, :last_updated, :pending, :interaction_count)\n          '
    db.engine.execute(text(sql), **params)

def add_journalist():
    if False:
        while True:
            i = 10
    if random_bool():
        otp_secret = random_chars(16, string.ascii_uppercase + '234567')
    else:
        otp_secret = None
    is_totp = random_bool()
    if is_totp:
        hotp_counter = 0 if random_bool() else None
    else:
        hotp_counter = random.randint(0, 10000) if random_bool() else None
    last_token = random_chars(6, string.digits) if random_bool() else None
    params = {'username': random_username(), 'pw_salt': random_bytes(1, 64, nullable=True), 'pw_hash': random_bytes(32, 64, nullable=True), 'is_admin': bool_or_none(), 'otp_secret': otp_secret, 'is_totp': is_totp, 'hotp_counter': hotp_counter, 'last_token': last_token, 'created_on': random_datetime(nullable=True), 'last_access': random_datetime(nullable=True), 'passphrase_hash': random_bytes(32, 64, nullable=True)}
    sql = 'INSERT INTO journalists (username, pw_salt, pw_hash,\n                is_admin, otp_secret, is_totp, hotp_counter, last_token,\n                created_on, last_access, passphrase_hash)\n             VALUES (:username, :pw_salt, :pw_hash, :is_admin,\n                :otp_secret, :is_totp, :hotp_counter, :last_token,\n                :created_on, :last_access, :passphrase_hash);\n          '
    db.engine.execute(text(sql), **params)

class UpgradeTester:
    """This migration verifies that the deleted_by_source column now exists,
    and that the data migration completed successfully.
    """
    SOURCE_NUM = 200
    JOURNO_NUM = 20

    def __init__(self, config):
        if False:
            print('Hello World!')
        self.config = config
        self.app = create_app(config)

    def load_data(self):
        if False:
            return 10
        with self.app.app_context():
            for _ in range(self.JOURNO_NUM):
                add_journalist()
            add_source()
            for jid in range(1, self.JOURNO_NUM):
                self.add_reply(jid, 1)
            db.session.commit()

    @staticmethod
    def add_reply(journalist_id, source_id):
        if False:
            i = 10
            return i + 15
        params = {'journalist_id': journalist_id, 'source_id': source_id, 'filename': random_chars(50), 'size': random.randint(0, 1024 * 1024 * 500), 'deleted_by_source': False}
        sql = 'INSERT INTO replies (journalist_id, source_id, filename,\n                    size, deleted_by_source)\n                 VALUES (:journalist_id, :source_id, :filename, :size,\n                         :deleted_by_source)\n              '
        db.engine.execute(text(sql), **params)

    def check_upgrade(self):
        if False:
            i = 10
            return i + 15
        with self.app.app_context():
            replies = db.engine.execute(text('SELECT * FROM replies')).fetchall()
            assert len(replies) == self.JOURNO_NUM - 1
            for reply in replies:
                assert reply.uuid is not None

class DowngradeTester:
    SOURCE_NUM = 200
    JOURNO_NUM = 20

    def __init__(self, config):
        if False:
            while True:
                i = 10
        self.config = config
        self.app = create_app(config)

    def load_data(self):
        if False:
            while True:
                i = 10
        with self.app.app_context():
            for _ in range(self.JOURNO_NUM):
                add_journalist()
            add_source()
            for jid in range(1, self.JOURNO_NUM):
                self.add_reply(jid, 1)
            db.session.commit()

    @staticmethod
    def add_reply(journalist_id, source_id):
        if False:
            for i in range(10):
                print('nop')
        params = {'journalist_id': journalist_id, 'source_id': source_id, 'uuid': str(uuid.uuid4()), 'filename': random_chars(50), 'size': random.randint(0, 1024 * 1024 * 500), 'deleted_by_source': False}
        sql = 'INSERT INTO replies (journalist_id, source_id, uuid, filename,\n                    size, deleted_by_source)\n                 VALUES (:journalist_id, :source_id, :uuid, :filename, :size,\n                    :deleted_by_source)\n              '
        db.engine.execute(text(sql), **params)

    def check_downgrade(self):
        if False:
            i = 10
            return i + 15
        'Verify that the deleted_by_source column is now gone, and\n        otherwise the table has the expected number of rows.\n        '
        with self.app.app_context():
            sql = 'SELECT * FROM replies'
            replies = db.engine.execute(text(sql)).fetchall()
            for reply in replies:
                try:
                    assert reply['uuid'] is None
                except NoSuchColumnError:
                    pass
            assert len(replies) == self.JOURNO_NUM - 1