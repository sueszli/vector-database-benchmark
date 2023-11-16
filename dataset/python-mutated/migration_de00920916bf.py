import random
import uuid
from db import db
from journalist_app import create_app
from sqlalchemy import text
from .helpers import random_chars
random.seed('くコ:彡')

class Helper:

    def __init__(self):
        if False:
            return 10
        self.journalist_id = None

    def create_journalist(self):
        if False:
            print('Hello World!')
        if self.journalist_id is not None:
            raise RuntimeError('Journalist already created')
        params = {'uuid': str(uuid.uuid4()), 'username': random_chars(50), 'session_nonce': 0, 'otp_secret': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ234567'}
        sql = 'INSERT INTO journalists (uuid, username, otp_secret, session_nonce)\n                 VALUES (:uuid, :username, :otp_secret, :session_nonce)\n              '
        self.journalist_id = db.engine.execute(text(sql), **params).lastrowid

class UpgradeTester(Helper):
    """
    Checks schema to verify that the otp_secret varchar "length" has been updated.
    Varchar specified length isn't enforced by sqlite but it's good to verify that
    the migration worked as expected.
    """

    def __init__(self, config):
        if False:
            print('Hello World!')
        Helper.__init__(self)
        self.config = config
        self.app = create_app(config)

    def load_data(self):
        if False:
            return 10
        with self.app.app_context():
            self.create_journalist()

    def check_upgrade(self):
        if False:
            for i in range(10):
                print('nop')
        with self.app.app_context():
            journalists_sql = 'SELECT * FROM journalists'
            journalist = db.engine.execute(text(journalists_sql)).first()
            assert len(journalist['otp_secret']) == 32

class DowngradeTester(Helper):

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        Helper.__init__(self)
        self.config = config
        self.app = create_app(config)

    def load_data(self):
        if False:
            i = 10
            return i + 15
        with self.app.app_context():
            self.create_journalist()

    def check_downgrade(self):
        if False:
            return 10
        with self.app.app_context():
            journalists_sql = 'SELECT * FROM journalists'
            journalist = db.engine.execute(text(journalists_sql)).first()
            assert len(journalist['otp_secret']) == 32