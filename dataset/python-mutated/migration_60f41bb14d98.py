import random
import string
import uuid
from db import db
from journalist_app import create_app
from sqlalchemy import text
from sqlalchemy.exc import NoSuchColumnError
from .helpers import bool_or_none, random_bool, random_bytes, random_chars, random_datetime, random_name, random_username
random.seed('ᕕ( ᐛ )ᕗ')

class UpgradeTester:
    """This migration verifies that the session_nonce column now exists, and
    that the data migration completed successfully.
    """
    JOURNO_NUM = 20

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
            for _ in range(self.JOURNO_NUM):
                self.add_journalist()
            db.session.commit()

    @staticmethod
    def add_journalist():
        if False:
            for i in range(10):
                print('nop')
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
        params = {'username': random_username(), 'uuid': str(uuid.uuid4()), 'first_name': random_name(), 'last_name': random_name(), 'pw_salt': random_bytes(1, 64, nullable=True), 'pw_hash': random_bytes(32, 64, nullable=True), 'is_admin': bool_or_none(), 'otp_secret': otp_secret, 'is_totp': is_totp, 'hotp_counter': hotp_counter, 'last_token': last_token, 'created_on': random_datetime(nullable=True), 'last_access': random_datetime(nullable=True), 'passphrase_hash': random_bytes(32, 64, nullable=True)}
        sql = 'INSERT INTO journalists (username, uuid, first_name, last_name,\n        pw_salt, pw_hash, is_admin, otp_secret, is_totp, hotp_counter,\n        last_token, created_on, last_access, passphrase_hash)\n                 VALUES (:username, :uuid, :first_name, :last_name, :pw_salt,\n                    :pw_hash, :is_admin, :otp_secret, :is_totp, :hotp_counter,\n                    :last_token, :created_on, :last_access, :passphrase_hash);\n              '
        db.engine.execute(text(sql), **params)

    def check_upgrade(self):
        if False:
            for i in range(10):
                print('nop')
        with self.app.app_context():
            journalists = db.engine.execute(text('SELECT * FROM journalists')).fetchall()
            for journalist in journalists:
                assert journalist.session_nonce is not None

class DowngradeTester:
    JOURNO_NUM = 20

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
            db.session.commit()

    @staticmethod
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
        params = {'username': random_username(), 'uuid': str(uuid.uuid4()), 'first_name': random_name(), 'last_name': random_name(), 'pw_salt': random_bytes(1, 64, nullable=True), 'pw_hash': random_bytes(32, 64, nullable=True), 'is_admin': bool_or_none(), 'session_nonce': random.randint(0, 10000), 'otp_secret': otp_secret, 'is_totp': is_totp, 'hotp_counter': hotp_counter, 'last_token': last_token, 'created_on': random_datetime(nullable=True), 'last_access': random_datetime(nullable=True), 'passphrase_hash': random_bytes(32, 64, nullable=True)}
        sql = 'INSERT INTO journalists (username, uuid, first_name, last_name,\n        pw_salt, pw_hash, is_admin, session_nonce, otp_secret, is_totp,\n        hotp_counter, last_token, created_on, last_access, passphrase_hash)\n                 VALUES (:username, :uuid, :first_name, :last_name, :pw_salt,\n                    :pw_hash, :is_admin, :session_nonce, :otp_secret, :is_totp,\n                    :hotp_counter, :last_token, :created_on, :last_access,\n                    :passphrase_hash);\n              '
        db.engine.execute(text(sql), **params)

    def check_downgrade(self):
        if False:
            for i in range(10):
                print('nop')
        'Verify that the session_nonce column is now gone, but otherwise the\n        table has the expected number of rows.\n        '
        with self.app.app_context():
            sql = 'SELECT * FROM journalists'
            journalists = db.engine.execute(text(sql)).fetchall()
            for journalist in journalists:
                try:
                    assert journalist['session_nonce'] is None
                except NoSuchColumnError:
                    pass