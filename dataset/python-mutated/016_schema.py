import peewee as pw
SCHEMA_VERSION = 16

def migrate(migrator, database, fake=False, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    migrator.add_fields('income', settled_ts=pw.IntegerField(null=True))

def rollback(migrator, database, fake=False, **kwargs):
    if False:
        print('Hello World!')
    migrator.remove_fields('income', 'settled_ts')