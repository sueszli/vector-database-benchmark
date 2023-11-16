import datetime as dt
import peewee as pw
SCHEMA_VERSION = 44

def migrate(migrator, database, fake=False, **kwargs):
    if False:
        while True:
            i = 10

    @migrator.create_model
    class AppConfiguration(pw.Model):
        created_date = pw.UTCDateTimeField(default=dt.datetime.now)
        modified_date = pw.UTCDateTimeField(default=dt.datetime.now)
        app_id = pw.CharField(primary_key=True, max_length=255)
        enabled = pw.BooleanField(default=False)

        class Meta:
            db_table = 'appconfiguration'

    @migrator.create_model
    class EnvConfiguration(pw.Model):
        created_date = pw.UTCDateTimeField(default=dt.datetime.now)
        modified_date = pw.UTCDateTimeField(default=dt.datetime.now)
        env_id = pw.CharField(primary_key=True, max_length=255)
        enabled = pw.BooleanField(default=False)

        class Meta:
            db_table = 'envconfiguration'

def rollback(migrator, database, fake=False, **kwargs):
    if False:
        print('Hello World!')
    migrator.remove_model('envconfiguration')
    migrator.remove_model('appconfiguration')