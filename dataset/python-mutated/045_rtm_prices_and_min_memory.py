import peewee as pw
SCHEMA_VERSION = 45

def migrate(migrator, database, fake=False, **kwargs):
    if False:
        return 10
    migrator.add_fields('requestedtask', min_memory=pw.IntegerField(default=0))
    migrator.change_fields('requestedtask', max_price_per_hour=pw.HexIntegerField())
    migrator.change_fields('requestedsubtask', price=pw.HexIntegerField(null=True))

def rollback(migrator, database, fake=False, **kwargs):
    if False:
        print('Hello World!')
    migrator.change_fields('requestedsubtask', price=pw.IntegerField(null=True))
    migrator.change_fields('requestedtask', max_price_per_hour=pw.IntegerField())
    migrator.remove_fields('requestedtask', 'min_memory')