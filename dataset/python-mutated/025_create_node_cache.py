import datetime
import peewee as pw
SCHEMA_VERSION = 25

def migrate(migrator, database, fake=False, **kwargs):
    if False:
        for i in range(10):
            print('nop')

    @migrator.create_model
    class CachedNode(pw.Model):
        node = pw.CharField(unique=True)
        node_field = pw.NodeField()
        created_date = pw.DateTimeField(default=datetime.datetime.now)
        modified_date = pw.DateTimeField(default=datetime.datetime.now)

        class Meta:
            db_table = 'cachednode'

def rollback(migrator, database, fake=False, **kwargs):
    if False:
        i = 10
        return i + 15
    migrator.remove_model('cachednode')