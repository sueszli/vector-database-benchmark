import peewee as pw
SCHEMA_VERSION = 15

def migrate(migrator, *_, **__):
    if False:
        return 10
    migrator.add_fields('performance', min_accepted_step=pw.FloatField(default=300.0))

def rollback(migrator, *_, **__):
    if False:
        i = 10
        return i + 15
    migrator.remove_fields('performance', 'min_accepted_step')