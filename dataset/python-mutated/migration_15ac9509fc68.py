import random
import string
from db import db
from journalist_app import create_app
from sqlalchemy import text
from .helpers import bool_or_none, random_bool, random_bytes, random_chars, random_datetime, random_username
random.seed('ᕕ( ᐛ )ᕗ')

class UpgradeTester:
    """This migration has no upgrade because there are no tables in the
    database prior to running, so there is no data to load or test.
    """

    def __init__(self, config):
        if False:
            print('Hello World!')
        pass

    def load_data(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def check_upgrade(self):
        if False:
            i = 10
            return i + 15
        pass

class DowngradeTester:
    JOURNO_NUM = 200
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
            for _ in range(self.JOURNO_NUM):
                self.add_journalist()
            for _ in range(self.SOURCE_NUM):
                self.add_source()
            for jid in range(1, self.JOURNO_NUM, 10):
                for _ in range(random.randint(1, 3)):
                    self.add_journalist_login_attempt(jid)
            for jid in range(1, self.JOURNO_NUM, 10):
                for sid in range(1, self.SOURCE_NUM, 10):
                    self.add_reply(jid, sid)
            for sid in range(1, self.SOURCE_NUM, 10):
                self.add_source_star(sid)
            for sid in range(1, self.SOURCE_NUM, 8):
                for _ in range(random.randint(1, 3)):
                    self.add_submission(sid)
            for sid in range(self.SOURCE_NUM, self.SOURCE_NUM + 50):
                self.add_submission(sid)
            db.session.commit()

    @staticmethod
    def add_journalist():
        if False:
            print('Hello World!')
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
        params = {'username': random_username(), 'pw_salt': random_bytes(1, 64, nullable=True), 'pw_hash': random_bytes(32, 64, nullable=True), 'is_admin': bool_or_none(), 'otp_secret': otp_secret, 'is_totp': is_totp, 'hotp_counter': hotp_counter, 'last_token': last_token, 'created_on': random_datetime(nullable=True), 'last_access': random_datetime(nullable=True)}
        sql = 'INSERT INTO journalists (username, pw_salt, pw_hash,\n                    is_admin, otp_secret, is_totp, hotp_counter, last_token,\n                    created_on, last_access)\n                 VALUES (:username, :pw_salt, :pw_hash, :is_admin,\n                    :otp_secret, :is_totp, :hotp_counter, :last_token,\n                    :created_on, :last_access);\n              '
        db.engine.execute(text(sql), **params)

    @staticmethod
    def add_source():
        if False:
            print('Hello World!')
        filesystem_id = random_chars(96) if random_bool() else None
        params = {'filesystem_id': filesystem_id, 'journalist_designation': random_chars(50), 'flagged': bool_or_none(), 'last_updated': random_datetime(nullable=True), 'pending': bool_or_none(), 'interaction_count': random.randint(0, 1000)}
        sql = 'INSERT INTO sources (filesystem_id, journalist_designation,\n                    flagged, last_updated, pending, interaction_count)\n                 VALUES (:filesystem_id, :journalist_designation, :flagged,\n                    :last_updated, :pending, :interaction_count)\n              '
        db.engine.execute(text(sql), **params)

    @staticmethod
    def add_journalist_login_attempt(journalist_id):
        if False:
            return 10
        params = {'timestamp': random_datetime(nullable=True), 'journalist_id': journalist_id}
        sql = 'INSERT INTO journalist_login_attempt (timestamp,\n                    journalist_id)\n                 VALUES (:timestamp, :journalist_id)\n              '
        db.engine.execute(text(sql), **params)

    @staticmethod
    def add_reply(journalist_id, source_id):
        if False:
            i = 10
            return i + 15
        params = {'journalist_id': journalist_id, 'source_id': source_id, 'filename': random_chars(50), 'size': random.randint(0, 1024 * 1024 * 500)}
        sql = 'INSERT INTO replies (journalist_id, source_id, filename,\n                    size)\n                 VALUES (:journalist_id, :source_id, :filename, :size)\n              '
        db.engine.execute(text(sql), **params)

    @staticmethod
    def add_source_star(source_id):
        if False:
            print('Hello World!')
        params = {'source_id': source_id, 'starred': bool_or_none()}
        sql = 'INSERT INTO source_stars (source_id, starred)\n                 VALUES (:source_id, :starred)\n              '
        db.engine.execute(text(sql), **params)

    @staticmethod
    def add_submission(source_id):
        if False:
            print('Hello World!')
        params = {'source_id': source_id, 'filename': random_chars(50), 'size': random.randint(0, 1024 * 1024 * 500), 'downloaded': bool_or_none()}
        sql = 'INSERT INTO submissions (source_id, filename, size,\n                    downloaded)\n                 VALUES (:source_id, :filename, :size, :downloaded)\n              '
        db.engine.execute(text(sql), **params)

    def check_downgrade(self):
        if False:
            print('Hello World!')
        "We don't need to check anything on this downgrade because the\n        migration drops all the tables. Thus, there is nothing to do.\n        "