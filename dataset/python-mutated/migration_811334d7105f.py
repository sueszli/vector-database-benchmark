import uuid
from db import db
from journalist_app import create_app
from sqlalchemy import text

class UpgradeTester:

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        self.config = config
        self.app = create_app(self.config)
        self.uuid = str(uuid.uuid4())

    def load_data(self):
        if False:
            i = 10
            return i + 15
        'Create a source'
        with self.app.app_context():
            source = {'uuid': self.uuid, 'filesystem_id': '5678', 'journalist_designation': 'alienated licensee', 'interaction_count': 0}
            sql = '                INSERT INTO sources (uuid, filesystem_id, journalist_designation,\n                    interaction_count)\n                VALUES (:uuid, :filesystem_id, :journalist_designation,\n                    :interaction_count)'
            db.engine.execute(text(sql), **source)

    def check_upgrade(self):
        if False:
            for i in range(10):
                print('nop')
        'Verify PGP fields can be queried and modified'
        with self.app.app_context():
            query_sql = '            SELECT pgp_fingerprint, pgp_public_key, pgp_secret_key\n            FROM sources\n            WHERE uuid = :uuid'
            source = db.engine.execute(text(query_sql), uuid=self.uuid).fetchone()
            assert source == (None, None, None)
            update_sql = '            UPDATE sources\n            SET pgp_fingerprint=:pgp_fingerprint, pgp_public_key=:pgp_public_key,\n                pgp_secret_key=:pgp_secret_key\n            WHERE uuid = :uuid'
            db.engine.execute(text(update_sql), pgp_fingerprint='AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA', pgp_public_key='a public key!', pgp_secret_key='a secret key!', uuid=self.uuid)
            source = db.engine.execute(text(query_sql), uuid=self.uuid).fetchone()
            assert source == ('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA', 'a public key!', 'a secret key!')

class DowngradeTester:

    def __init__(self, config):
        if False:
            for i in range(10):
                print('nop')
        self.config = config
        self.app = create_app(self.config)
        self.uuid = str(uuid.uuid4())

    def load_data(self):
        if False:
            while True:
                i = 10
        'Create a source with a PGP key pair stored'
        with self.app.app_context():
            source = {'uuid': self.uuid, 'filesystem_id': '1234', 'journalist_designation': 'mucky pine', 'interaction_count': 0, 'pgp_fingerprint': 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA', 'pgp_public_key': 'very public', 'pgp_secret_key': 'very secret'}
            sql = '                INSERT INTO sources (uuid, filesystem_id, journalist_designation,\n                    interaction_count, pgp_fingerprint, pgp_public_key, pgp_secret_key)\n                VALUES (:uuid, :filesystem_id, :journalist_designation,\n                    :interaction_count, :pgp_fingerprint, :pgp_public_key, :pgp_secret_key)'
            db.engine.execute(text(sql), **source)

    def check_downgrade(self):
        if False:
            i = 10
            return i + 15
        'Verify the downgrade does nothing, i.e. the PGP fields are still there'
        with self.app.app_context():
            sql = '            SELECT pgp_fingerprint, pgp_public_key, pgp_secret_key\n            FROM sources\n            WHERE uuid = :uuid'
            source = db.engine.execute(text(sql), uuid=self.uuid).fetchone()
            print(source)
            assert source == ('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA', 'very public', 'very secret')