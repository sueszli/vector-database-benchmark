import peewee as pw
SCHEMA_VERSION = 11

def migrate(migrator, _database, **_kwargs):
    if False:
        while True:
            i = 10
    'Write your migrations here.'
    migrator.remove_fields('expectedincome', 'sender_node_details')

def rollback(migrator, _database, **_kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Write your rollback migrations here.'
    migrator.add_fields('expectedincome', sender_node_details=pw.NodeField())