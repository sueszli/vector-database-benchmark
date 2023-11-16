import datetime
import peewee as pw
SCHEMA_VERSION = 48

def migrate(migrator, database, fake=False, **kwargs):
    if False:
        return 10

    @migrator.create_model
    class Broadcast(pw.Model):
        timestamp = pw.IntegerField()
        broadcast_type = pw.IntegerField()
        signature = pw.BlobField()
        data = pw.BlobField()
        created_date = pw.DateTimeField(default=datetime.datetime.now)
        modified_date = pw.DateTimeField(default=datetime.datetime.now)

        class Meta:
            db_table = 'broadcast'

def rollback(migrator, database, fake=False, **kwargs):
    if False:
        while True:
            i = 10
    migrator.remove_model('broadcast')