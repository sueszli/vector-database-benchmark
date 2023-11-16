import unicodedata
from django.db import connection, migrations, models
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.migrations.state import StateApps
unicode_non_chars = {chr(x) for r in [range(64976, 65008), range(65534, 1114112, 65536), range(65535, 1114112, 65536)] for x in r}

def character_is_printable(character: str) -> bool:
    if False:
        while True:
            i = 10
    return not (unicodedata.category(character) in ['Cc', 'Cs'] or character in unicode_non_chars)

def fix_topics(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        print('Hello World!')
    Message = apps.get_model('zerver', 'Message')
    BATCH_SIZE = 10000
    messages_updated = 0
    lower_bound = 0
    max_id = Message.objects.aggregate(models.Max('id'))['id__max']
    if max_id is None:
        return
    print('')
    while lower_bound < max_id:
        print(f'Processed {lower_bound} / {max_id}')
        with connection.cursor() as cursor:
            cursor.execute('SELECT DISTINCT subject FROM zerver_message WHERE id > %s AND id <= %s', [lower_bound, lower_bound + BATCH_SIZE])
            results = cursor.fetchall()
            topics = [r[0] for r in results]
            for topic in topics:
                fixed_topic = ''.join((character for character in topic if character_is_printable(character)))
                if fixed_topic == topic:
                    continue
                if fixed_topic == '':
                    fixed_topic = '(no topic)'
                cursor.execute('UPDATE zerver_message SET subject = %s WHERE subject = %s AND id > %s AND id <= %s', [fixed_topic, topic, lower_bound, lower_bound + BATCH_SIZE])
                messages_updated += cursor.rowcount
            lower_bound += BATCH_SIZE
    if messages_updated > 0:
        print(f'Fixed invalid topics for {messages_updated} messages.')

class Migration(migrations.Migration):
    atomic = False
    dependencies = [('zerver', '0370_realm_enable_spectator_access')]
    operations = [migrations.RunPython(fix_topics, reverse_code=migrations.RunPython.noop)]