import peewee as pw
SCHEMA_VERSION = 39

def migrate(migrator, database, fake=False, **kwargs):
    if False:
        i = 10
        return i + 15
    migrator.add_fields('performance', cpu_usage=pw.IntegerField(default=0))

def rollback(migrator, database, fake=False, **kwargs):
    if False:
        i = 10
        return i + 15
    migrator.remove_fields('performance', 'cpu_usage')