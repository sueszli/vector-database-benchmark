from django.db import migrations
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.migrations.state import StateApps

def backfill_first_message_id(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        print('Hello World!')
    Stream = apps.get_model('zerver', 'Stream')
    Message = apps.get_model('zerver', 'Message')
    for stream in Stream.objects.all():
        first_message = Message.objects.filter(recipient__type_id=stream.id, recipient__type=2).first()
        if first_message is None:
            continue
        stream.first_message_id = first_message.id
        stream.save()

class Migration(migrations.Migration):
    dependencies = [('zerver', '0209_stream_first_message_id')]
    operations = [migrations.RunPython(backfill_first_message_id, reverse_code=migrations.RunPython.noop, elidable=True)]