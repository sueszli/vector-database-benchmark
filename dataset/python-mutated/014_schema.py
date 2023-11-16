import peewee as pw
SCHEMA_VERSION = 14

def migrate(migrator, *_, **__):
    if False:
        i = 10
        return i + 15
    migrator.add_fields('income', overdue=pw.BooleanField(default=False))

def rollback(migrator, *_, **__):
    if False:
        i = 10
        return i + 15
    migrator.remove_fields('income', 'overdue')