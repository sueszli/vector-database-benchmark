import datetime
import logging
import peewee as pw
SCHEMA_VERSION = 34
logger = logging.getLogger('golem.database')

def migrate(migrator, database, fake=False, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    migrator.remove_model('payment')

def rollback(migrator, database, fake=False, **kwargs):
    if False:
        while True:
            i = 10

    @migrator.create_model
    class Payment(pw.Model):
        subtask = pw.CharField()
        status = pw.IntegerField()
        payee = pw.RawCharField()
        value = pw.CharField()
        details = pw.RawCharField()
        processed_ts = pw.IntegerField(null=True)
        created_date = pw.DateTimeField(default=datetime.datetime.now)
        modified_date = pw.DateTimeField(default=datetime.datetime.now)

        class Meta:
            db_table = 'payment'