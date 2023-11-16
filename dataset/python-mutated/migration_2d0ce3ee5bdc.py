import random
import string
from uuid import uuid4
from db import db
from journalist_app import create_app
from sqlalchemy import text
from .helpers import bool_or_none, random_bool, random_bytes, random_chars, random_datetime, random_username
random.seed('ᕕ( ᐛ )ᕗ')

class Helper:

    @staticmethod
    def add_source():
        if False:
            return 10
        filesystem_id = random_chars(96) if random_bool() else None
        params = {'uuid': str(uuid4()), 'filesystem_id': filesystem_id, 'journalist_designation': random_chars(50), 'flagged': bool_or_none(), 'last_updated': random_datetime(nullable=True), 'pending': bool_or_none(), 'interaction_count': random.randint(0, 1000)}
        sql = 'INSERT INTO sources (uuid, filesystem_id,\n                    journalist_designation, flagged, last_updated, pending,\n                    interaction_count)\n                 VALUES (:uuid, :filesystem_id, :journalist_designation,\n                    :flagged, :last_updated, :pending, :interaction_count)\n              '
        db.engine.execute(text(sql), **params)

    @staticmethod
    def add_journalist_login_attempt(journalist_id):
        if False:
            i = 10
            return i + 15
        params = {'timestamp': random_datetime(nullable=True), 'journalist_id': journalist_id}
        sql = 'INSERT INTO journalist_login_attempt (timestamp,\n                    journalist_id)\n                 VALUES (:timestamp, :journalist_id)\n              '
        db.engine.execute(text(sql), **params)

    @staticmethod
    def add_reply(journalist_id, source_id):
        if False:
            while True:
                i = 10
        params = {'journalist_id': journalist_id, 'source_id': source_id, 'filename': random_chars(50), 'size': random.randint(0, 1024 * 1024 * 500)}
        sql = 'INSERT INTO replies (journalist_id, source_id, filename,\n                    size)\n                 VALUES (:journalist_id, :source_id, :filename, :size)\n              '
        db.engine.execute(text(sql), **params)

    @staticmethod
    def extract(app):
        if False:
            for i in range(10):
                print('nop')
        with app.app_context():
            sql = 'SELECT j.id, count(distinct a.id), count(distinct r.id)\n                     FROM journalists AS j\n                     LEFT OUTER JOIN journalist_login_attempt AS a\n                     ON a.journalist_id = j.id\n                     LEFT OUTER JOIN replies AS r\n                     ON r.journalist_id = j.id\n                     GROUP BY j.id\n                     ORDER BY j.id\n                  '
            return list(db.session.execute(text(sql)))

class UpgradeTester(Helper):
    JOURNO_NUM = 100

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        self.config = config
        self.app = create_app(config)
        self.initial_data = None

    def load_data(self):
        if False:
            for i in range(10):
                print('nop')
        with self.app.app_context():
            for _ in range(self.JOURNO_NUM):
                self.add_journalist()
            self.add_source()
            for jid in range(1, self.JOURNO_NUM):
                for _ in range(random.randint(1, 3)):
                    self.add_journalist_login_attempt(jid)
            for jid in range(1, self.JOURNO_NUM):
                self.add_reply(jid, 1)
            db.session.commit()
            self.initial_data = self.extract(self.app)

    def check_upgrade(self):
        if False:
            i = 10
            return i + 15
        extracted = self.extract(self.app)
        assert len(extracted) == self.JOURNO_NUM
        assert extracted == self.initial_data

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

class DowngradeTester(Helper):
    JOURNO_NUM = 100

    def __init__(self, config):
        if False:
            print('Hello World!')
        self.config = config
        self.app = create_app(config)
        self.initial_data = None

    def load_data(self):
        if False:
            for i in range(10):
                print('nop')
        with self.app.app_context():
            for _ in range(self.JOURNO_NUM):
                self.add_journalist()
            self.add_source()
            for jid in range(1, self.JOURNO_NUM):
                for _ in range(random.randint(1, 3)):
                    self.add_journalist_login_attempt(jid)
            for jid in range(1, self.JOURNO_NUM):
                self.add_reply(jid, 1)
            db.session.commit()
            self.initial_data = self.extract(self.app)

    def check_downgrade(self):
        if False:
            i = 10
            return i + 15
        extracted = self.extract(self.app)
        assert len(extracted) == self.JOURNO_NUM
        assert extracted == self.initial_data

    @staticmethod
    def add_journalist():
        if False:
            i = 10
            return i + 15
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
        sql = 'INSERT INTO journalists (username, pw_salt, pw_hash,\n                    is_admin, otp_secret, is_totp, hotp_counter, last_token,\n                    created_on, last_access, passphrase_hash)\n                 VALUES (:username, :pw_salt, :pw_hash, :is_admin,\n                    :otp_secret, :is_totp, :hotp_counter, :last_token,\n                    :created_on, :last_access, :passphrase_hash);\n              '
        db.engine.execute(text(sql), **params)