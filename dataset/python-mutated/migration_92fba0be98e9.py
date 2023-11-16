import pytest
import sqlalchemy
from db import db
from journalist_app import create_app
from .helpers import random_bool, random_datetime

class UpgradeTester:

    def __init__(self, config):
        if False:
            return 10
        self.config = config
        self.app = create_app(config)

    def load_data(self):
        if False:
            print('Hello World!')
        with self.app.app_context():
            self.update_config()
            db.session.commit()

    @staticmethod
    def update_config():
        if False:
            return 10
        params = {'valid_until': random_datetime(nullable=True), 'allow_document_uploads': random_bool()}
        sql = '\n        INSERT INTO instance_config (\n            valid_until, allow_document_uploads\n        ) VALUES (\n            :valid_until, :allow_document_uploads\n        )\n        '
        db.engine.execute(sqlalchemy.text(sql), **params)

    def check_upgrade(self):
        if False:
            print('Hello World!')
        "\n        Check the new `organization_name` column\n\n        Querying `organization_name` shouldn't cause an error, but it should not yet be set.\n        "
        with self.app.app_context():
            configs = db.engine.execute(sqlalchemy.text('SELECT * FROM instance_config WHERE organization_name IS NOT NULL')).fetchall()
            assert len(configs) == 0

class DowngradeTester:

    def __init__(self, config):
        if False:
            return 10
        self.config = config
        self.app = create_app(config)

    def load_data(self):
        if False:
            return 10
        pass

    def check_downgrade(self):
        if False:
            while True:
                i = 10
        '\n        After downgrade, using `organization_name` in a query should raise an exception\n        '
        with self.app.app_context(), pytest.raises(sqlalchemy.exc.OperationalError):
            db.engine.execute(sqlalchemy.text('SELECT * FROM instance_config WHERE organization_name IS NOT NULL')).fetchall()