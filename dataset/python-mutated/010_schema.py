import peewee as pw
SCHEMA_VERSION = 10

def migrate(migrator, _database, **_kwargs):
    if False:
        return 10
    'Write your migrations here.'
    migrator.remove_fields('expectedincome', 'task')
    migrator.remove_fields('income', 'task')
    migrator.remove_fields('income', 'block_number')

def rollback(migrator, _database, **_kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Write your rollback migrations here.'
    migrator.add_fields('expectedincome', task=pw.CharField(max_length=255))
    migrator.add_fields('income', task=pw.CharField(max_length=255))
    migrator.add_fields('income', block_number=pw.HexIntegerField())