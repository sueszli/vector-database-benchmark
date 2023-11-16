import uuid
from db import db
from journalist_app import create_app
from sqlalchemy import text
FILESYSTEM_ID = 'FPLIUY3FWFROQ52YHDMXFYKOTIK2FT4GAFN6HTCPPG3TSNAHYOPDDT5C3TRJWDV2IL3JDOS4NFAJNEI73KRQ7HQEVNAF35UCCW5M7VI='

class UpgradeTester:
    """Insert a source, verify the filesystem_id makes it through untouched"""

    def __init__(self, config):
        if False:
            while True:
                i = 10
        'This function MUST accept an argument named `config`.\n        You will likely want to save a reference to the config in your\n        class, so you can access the database later.\n        '
        self.config = config
        self.app = create_app(config)

    def load_data(self):
        if False:
            for i in range(10):
                print('nop')
        'This function loads data into the database and filesystem. It is\n        executed before the upgrade.\n        '
        with self.app.app_context():
            sources = [{'uuid': str(uuid.uuid4()), 'filesystem_id': FILESYSTEM_ID, 'journalist_designation': 'sunburned arraignment', 'interaction_count': 0}, {'uuid': str(uuid.uuid4()), 'filesystem_id': None, 'journalist_designation': 'needy transponder', 'interaction_count': 0}]
            sql = '                INSERT INTO sources (uuid, filesystem_id, journalist_designation, interaction_count)\n                VALUES (:uuid, :filesystem_id, :journalist_designation, :interaction_count)'
            for params in sources:
                db.engine.execute(text(sql), **params)
            for source_id in (1, 2):
                db.engine.execute(text('INSERT INTO source_stars (source_id, starred) VALUES (:source_id, :starred)'), {'source_id': source_id, 'starred': True})

    def check_upgrade(self):
        if False:
            print('Hello World!')
        'This function is run after the upgrade and verifies the state\n        of the database or filesystem. It MUST raise an exception if the\n        check fails.\n        '
        with self.app.app_context():
            sources = db.engine.execute('SELECT filesystem_id FROM sources').fetchall()
            assert len(sources) == 1
            assert sources[0][0] == FILESYSTEM_ID
            stars = db.engine.execute('SELECT source_id FROM source_stars').fetchall()
            assert stars == [(1,)]

class DowngradeTester:
    """Downgrading only makes fields nullable again, which is a
    non-destructive and safe operation"""

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        'This function MUST accept an argument named `config`.\n        You will likely want to save a reference to the config in your\n        class, so you can access the database later.\n        '
        self.config = config

    def load_data(self):
        if False:
            print('Hello World!')
        'This function loads data into the database and filesystem. It is\n        executed before the downgrade.\n        '

    def check_downgrade(self):
        if False:
            print('Hello World!')
        'This function is run after the downgrade and verifies the state\n        of the database or filesystem. It MUST raise an exception if the\n        check fails.\n        '