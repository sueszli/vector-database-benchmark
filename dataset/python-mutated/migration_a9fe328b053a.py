import random
import uuid
from db import db
from journalist_app import create_app
from sqlalchemy import text
from sqlalchemy.exc import NoSuchColumnError
from .helpers import random_chars
random.seed('⎦˚◡˚⎣')

class Helper:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.journalist_id = None

    def create_journalist(self):
        if False:
            while True:
                i = 10
        if self.journalist_id is not None:
            raise RuntimeError('Journalist already created')
        params = {'uuid': str(uuid.uuid4()), 'username': random_chars(50)}
        sql = 'INSERT INTO journalists (uuid, username)\n                 VALUES (:uuid, :username)\n              '
        self.journalist_id = db.engine.execute(text(sql), **params).lastrowid

    def create_journalist_after_migration(self):
        if False:
            i = 10
            return i + 15
        if self.journalist_id is not None:
            raise RuntimeError('Journalist already created')
        params = {'uuid': str(uuid.uuid4()), 'username': random_chars(50), 'first_name': random_chars(50), 'last_name': random_chars(50)}
        sql = '\n        INSERT INTO journalists (uuid, username, first_name, last_name)\n        VALUES (:uuid, :username, :first_name, :last_name)\n        '
        self.journalist_id = db.engine.execute(text(sql), **params).lastrowid

class UpgradeTester(Helper):

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
            while True:
                i = 10
        '\n        - Verify that Journalist first and last names are present after upgrade.\n        '
        with self.app.app_context():
            journalists_sql = 'SELECT * FROM journalists'
            journalists = db.engine.execute(text(journalists_sql)).fetchall()
            for journalist in journalists:
                assert journalist['first_name'] is None
                assert journalist['last_name'] is None

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
            self.create_journalist_after_migration()

    def check_downgrade(self):
        if False:
            print('Hello World!')
        '\n        - Verify that Journalist first and last names are gone after downgrade.\n        '
        with self.app.app_context():
            journalists_sql = 'SELECT * FROM journalists'
            journalists = db.engine.execute(text(journalists_sql)).fetchall()
            for journalist in journalists:
                try:
                    assert journalist['first_name']
                except NoSuchColumnError:
                    pass
                try:
                    assert journalist['last_name']
                except NoSuchColumnError:
                    pass