from django.contrib.postgres.operations import AddIndexConcurrently, RemoveIndexConcurrently
from django.db import connection, migrations, models
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.migrations.state import StateApps
from psycopg2.sql import SQL

def clear_old_data_for_unused_usermessage_flags(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        return 10
    "Because 'topic_wildcard_mentioned' and 'group_mentioned' flags are\n    reused flag slots (ref: c37871a) in the 'flags' bitfield, we're not\n    confident that their value is in 0 state on very old servers, and this\n    migration is to ensure that's the case.\n    Additionally, we are clearing 'force_expand' and 'force_collapse' unused\n    flags to save future work.\n    "
    with connection.cursor() as cursor:
        cursor.execute(SQL('SELECT MAX(id) FROM zerver_usermessage WHERE flags & 480 <> 0;'))
        (max_id,) = cursor.fetchone()
    if not max_id:
        return
    BATCH_SIZE = 5000
    lower_id_bound = 0
    while lower_id_bound < max_id:
        upper_id_bound = min(lower_id_bound + BATCH_SIZE, max_id)
        with connection.cursor() as cursor:
            query = SQL('\n                    UPDATE zerver_usermessage\n                    SET flags = (flags & ~(1 << 5) & ~(1 << 6) & ~(1 << 7) & ~(1 << 8))\n                    WHERE flags & 480 <> 0\n                    AND id > %(lower_id_bound)s AND id <= %(upper_id_bound)s;\n            ')
            cursor.execute(query, {'lower_id_bound': lower_id_bound, 'upper_id_bound': upper_id_bound})
        print(f'Processed {upper_id_bound} / {max_id}')
        lower_id_bound = lower_id_bound + BATCH_SIZE

class Migration(migrations.Migration):
    atomic = False
    dependencies = [('zerver', '0485_alter_usermessage_flags_and_add_index')]
    operations = [AddIndexConcurrently(model_name='usermessage', index=models.Index('id', condition=models.Q(('flags__andnz', 480)), name='zerver_usermessage_temp_clear_flags')), migrations.RunPython(clear_old_data_for_unused_usermessage_flags, reverse_code=migrations.RunPython.noop, elidable=True), RemoveIndexConcurrently(model_name='usermessage', name='zerver_usermessage_temp_clear_flags')]