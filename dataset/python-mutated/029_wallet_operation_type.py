import peewee as pw
SCHEMA_VERSION = 29

def migrate(migrator, database, fake=False, **kwargs):
    if False:
        while True:
            i = 10
    migrator.add_fields('walletoperation', operation_type=pw.CharField(default='task_payment'))

def rollback(migrator, database, fake=False, **kwargs):
    if False:
        print('Hello World!')
    migrator.remove_fields('walletoperation', 'task_payment')