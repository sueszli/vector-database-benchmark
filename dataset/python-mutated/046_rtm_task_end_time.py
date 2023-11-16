import peewee as pw
SCHEMA_VERSION = 46

def migrate(migrator, database, fake=False, **kwargs):
    if False:
        return 10
    migrator.add_fields('requestedtask', end_time=pw.UTCDateTimeField(null=True))

def rollback(migrator, database, fake=False, **kwargs):
    if False:
        i = 10
        return i + 15
    migrator.remove_fields('requestedtask', 'end_time')