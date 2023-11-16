import uuid
from db import db
from journalist_app import create_app
from sqlalchemy import text
from .helpers import random_datetime

class UpgradeTester:
    """Insert a Reply, SeenReply and JournalistLoginAttempt with journalist_id=NULL.
    Verify that the first two are reassociated to the "Deleted" user, while the last
    is deleted outright.
    """

    def __init__(self, config):
        if False:
            return 10
        'This function MUST accept an argument named `config`.\n        You will likely want to save a reference to the config in your\n        class, so you can access the database later.\n        '
        self.config = config
        self.app = create_app(config)

    def load_data(self):
        if False:
            for i in range(10):
                print('nop')
        'This function loads data into the database and filesystem. It is\n        executed before the upgrade.\n        '
        with self.app.app_context():
            params = {'uuid': str(uuid.uuid4()), 'journalist_id': None, 'source_id': 0, 'filename': 'dummy.txt', 'size': 1, 'checksum': '', 'deleted_by_source': False}
            sql = '                INSERT INTO replies (uuid, journalist_id, source_id, filename,\n                    size, checksum, deleted_by_source)\n                 VALUES (:uuid, :journalist_id, :source_id, :filename,\n                        :size, :checksum, :deleted_by_source);'
            db.engine.execute(text(sql), **params)
            for _ in range(2):
                db.engine.execute(text('                    INSERT INTO seen_replies (reply_id, journalist_id)\n                    VALUES (1, NULL);\n                    '))
            db.engine.execute(text('                INSERT INTO journalist_login_attempt (timestamp, journalist_id)\n                VALUES (:timestamp, NULL)\n                '), timestamp=random_datetime(nullable=False))

    def check_upgrade(self):
        if False:
            print('Hello World!')
        'This function is run after the upgrade and verifies the state\n        of the database or filesystem. It MUST raise an exception if the\n        check fails.\n        '
        with self.app.app_context():
            deleted = db.engine.execute('SELECT id, passphrase_hash, otp_secret FROM journalists WHERE username="deleted"').first()
            assert deleted[1].startswith('$argon2')
            assert len(deleted[2]) == 32
            deleted_id = deleted[0]
            replies = db.engine.execute(text('SELECT journalist_id FROM replies')).fetchall()
            assert len(replies) == 1
            assert replies[0][0] == deleted_id
            seen_replies = db.engine.execute(text('SELECT journalist_id FROM seen_replies')).fetchall()
            assert len(seen_replies) == 1
            assert seen_replies[0][0] == deleted_id
            login_attempts = db.engine.execute(text('SELECT * FROM journalist_login_attempt')).fetchall()
            assert login_attempts == []

class DowngradeTester:
    """Downgrading only makes fields nullable again, which is a
    non-destructive and safe operation"""

    def __init__(self, config):
        if False:
            return 10
        'This function MUST accept an argument named `config`.\n        You will likely want to save a reference to the config in your\n        class, so you can access the database later.\n        '
        self.config = config

    def load_data(self):
        if False:
            for i in range(10):
                print('nop')
        'This function loads data into the database and filesystem. It is\n        executed before the downgrade.\n        '

    def check_downgrade(self):
        if False:
            i = 10
            return i + 15
        'This function is run after the downgrade and verifies the state\n        of the database or filesystem. It MUST raise an exception if the\n        check fails.\n        '