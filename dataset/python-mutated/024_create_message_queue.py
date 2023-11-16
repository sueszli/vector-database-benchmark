import datetime
import peewee as pw
SCHEMA_VERSION = 21

def migrate(migrator, database, fake=False, **kwargs):
    if False:
        while True:
            i = 10

    @migrator.create_model
    class QueuedMessage(pw.Model):
        node = pw.CharField()
        msg_version = pw.CharField()
        msg_cls = pw.CharField()
        msg_data = pw.BlobField()
        created_date = pw.DateTimeField(default=datetime.datetime.now)
        modified_date = pw.DateTimeField(default=datetime.datetime.now)

        class Meta:
            db_table = 'queuedmessage'

def rollback(migrator, database, fake=False, **kwargs):
    if False:
        i = 10
        return i + 15
    migrator.remove_model('queuedmessage')