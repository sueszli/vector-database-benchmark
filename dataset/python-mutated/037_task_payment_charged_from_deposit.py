from golem import model
SCHEMA_VERSION = 37

def migrate(migrator, database, fake=False, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    migrator.add_fields('taskpayment', charged_from_deposit=model.BooleanField(null=True))

def rollback(migrator, database, fake=False, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    migrator.remove_fields('taskpayment', 'charged_from_deposit')