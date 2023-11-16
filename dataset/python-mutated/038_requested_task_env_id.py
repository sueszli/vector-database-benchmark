import peewee as pw
SCHEMA_VERSION = 38

def migrate(migrator, database, fake=False, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    migrator.add_fields('requestedtask', env_id=pw.CharField(max_length=255, null=True))

def rollback(migrator, database, fake=False, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    migrator.remove_fields('requestedtask', 'env_id')