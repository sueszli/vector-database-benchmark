import datetime as dt
import peewee as pw
SCHEMA_VERSION = 9

def migrate(migrator, _database, **_kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Write your migrations here.'

    @migrator.create_model
    class GenericKeyValue(pw.Model):
        key = pw.CharField(max_length=255, primary_key=True)
        created_date = pw.DateTimeField(default=dt.datetime.now)
        modified_date = pw.DateTimeField(default=dt.datetime.now)
        value = pw.CharField(max_length=255, null=True)

        class Meta:
            db_table = 'generickeyvalue'

def rollback(migrator, _database, **_kwargs):
    if False:
        while True:
            i = 10
    'Write your rollback migrations here.'
    migrator.remove_model('generickeyvalue')