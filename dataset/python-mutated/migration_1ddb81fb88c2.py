from db import db
from journalist_app import create_app
from sqlalchemy import text
index_definition = ('index', 'ix_one_active_instance_config', 'instance_config', 'CREATE UNIQUE INDEX ix_one_active_instance_config ON instance_config (valid_until IS NULL) WHERE valid_until IS NULL')

def get_schema(app):
    if False:
        for i in range(10):
            print('nop')
    with app.app_context():
        result = list(db.engine.execute(text('SELECT type, name, tbl_name, sql FROM sqlite_master')))
    return ((x[0], x[1], x[2], x[3]) for x in result)

class UpgradeTester:
    """
    Ensure that the new index is created.
    """

    def __init__(self, config):
        if False:
            while True:
                i = 10
        self.app = create_app(config)

    def load_data(self):
        if False:
            return 10
        pass

    def check_upgrade(self):
        if False:
            i = 10
            return i + 15
        schema = get_schema(self.app)
        print(schema)
        assert index_definition in schema

class DowngradeTester:
    """
    Ensure that the new index is removed.
    """

    def __init__(self, config):
        if False:
            return 10
        self.app = create_app(config)

    def load_data(self):
        if False:
            return 10
        pass

    def check_downgrade(self):
        if False:
            print('Hello World!')
        assert index_definition not in get_schema(self.app)