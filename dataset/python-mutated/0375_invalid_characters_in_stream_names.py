import unicodedata
from django.db import connection, migrations
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.migrations.state import StateApps
unicode_non_chars = {chr(x) for r in [range(64976, 65008), range(65534, 1114112, 65536), range(65535, 1114112, 65536)] for x in r}

def character_is_printable(character: str) -> bool:
    if False:
        return 10
    return not (unicodedata.category(character) in ['Cc', 'Cs'] or character in unicode_non_chars)

def fix_stream_names(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        while True:
            i = 10
    Stream = apps.get_model('zerver', 'Stream')
    Realm = apps.get_model('zerver', 'Realm')
    total_fixed_count = 0
    realm_ids = Realm.objects.values_list('id', flat=True)
    if len(realm_ids) == 0:
        return
    print('')
    for realm_id in realm_ids:
        print(f'Processing realm {realm_id}')
        realm_stream_dicts = Stream.objects.filter(realm_id=realm_id).values('id', 'name')
        occupied_stream_names = {stream_dict['name'] for stream_dict in realm_stream_dicts}
        for stream_dict in realm_stream_dicts:
            stream_name = stream_dict['name']
            fixed_stream_name = ''.join([character if character_is_printable(character) else '�' for character in stream_name])
            if fixed_stream_name == stream_name:
                continue
            if fixed_stream_name == '':
                fixed_stream_name = '(no name)'
            while fixed_stream_name in occupied_stream_names:
                fixed_stream_name += '_'
            occupied_stream_names.add(fixed_stream_name)
            total_fixed_count += 1
            with connection.cursor() as cursor:
                cursor.execute('UPDATE zerver_stream SET name = %s WHERE id = %s', [fixed_stream_name, stream_dict['id']])
    print(f'Fixed {total_fixed_count} stream names')

class Migration(migrations.Migration):
    atomic = False
    dependencies = [('zerver', '0374_backfill_user_delete_realmauditlog')]
    operations = [migrations.RunPython(fix_stream_names, reverse_code=migrations.RunPython.noop)]