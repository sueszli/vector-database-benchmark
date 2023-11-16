import datetime as dt
import peewee as pw
SCHEMA_VERSION = 40

def migrate(migrator, database, fake=False, **kwargs):
    if False:
        for i in range(10):
            print('nop')

    @migrator.create_model
    class QueuedVerification(pw.Model):
        created_date = pw.UTCDateTimeField(default=dt.datetime.now)
        modified_date = pw.UTCDateTimeField(default=dt.datetime.now)
        task_id = pw.CharField(max_length=255)
        subtask_id = pw.CharField(max_length=255)
        priority = pw.IntegerField(index=True, null=True)

        class Meta:
            db_table = 'queuedverification'
            primary_key = pw.CompositeKey('task_id', 'subtask_id')

def rollback(migrator, database, fake=False, **kwargs):
    if False:
        i = 10
        return i + 15
    migrator.remove_model('queuedverification')