SCHEMA_VERSION = 19

def migrate(migrator, database, fake=False, **kwargs):
    if False:
        while True:
            i = 10
    migrator.add_not_null('income', 'payer_address')

def rollback(migrator, database, fake=False, **kwargs):
    if False:
        i = 10
        return i + 15
    migrator.drop_not_null('income', 'payer_address')