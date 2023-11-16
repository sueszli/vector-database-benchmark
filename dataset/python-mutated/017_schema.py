import peewee as pw
SCHEMA_VERSION = 17

def migrate(migrator, database, fake=False, **kwargs):
    if False:
        return 10
    migrator.add_fields('knownhosts', metadata=pw.JsonField(default='{}'))

def rollback(migrator, database, fake=False, **kwargs):
    if False:
        return 10
    migrator.remove_fields('knownhosts', 'metadata')