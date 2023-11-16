import peewee as pw
SCHEMA_VERSION = 8

def migrate(migrator, _database, **_kwargs):
    if False:
        i = 10
        return i + 15
    'Write your migrations here.'
    migrator.add_fields('expectedincome', accepted_ts=pw.IntegerField(null=True))

def rollback(migrator, _database, **_kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Write your rollback migrations here.'
    migrator.remove_fields('expectedincome', 'accepted_ts')