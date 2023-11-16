from db import db
from journalist_app import create_app
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
instance_config_sql = 'SELECT * FROM instance_config'

class UpgradeTester:

    def __init__(self, config):
        if False:
            print('Hello World!')
        self.config = config
        self.app = create_app(config)

    def load_data(self):
        if False:
            return 10
        pass

    def check_upgrade(self):
        if False:
            i = 10
            return i + 15
        with self.app.app_context():
            db.engine.execute(text(instance_config_sql)).fetchall()

class DowngradeTester:

    def __init__(self, config):
        if False:
            print('Hello World!')
        self.config = config
        self.app = create_app(config)

    def load_data(self):
        if False:
            while True:
                i = 10
        pass

    def check_downgrade(self):
        if False:
            i = 10
            return i + 15
        with self.app.app_context():
            try:
                db.engine.execute(text(instance_config_sql)).fetchall()
            except OperationalError:
                pass