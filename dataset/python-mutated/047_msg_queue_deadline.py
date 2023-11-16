import peewee as pw
from golem.model import default_msg_deadline
SCHEMA_VERSION = 47

def migrate(migrator, database, fake=False, **kwargs):
    if False:
        return 10
    migrator.add_fields('queuedmessage', deadline=pw.UTCDateTimeField(default=default_msg_deadline()))

def rollback(migrator, database, fake=False, **kwargs):
    if False:
        i = 10
        return i + 15
    migrator.remove_fields('queuedmessage', 'deadline')