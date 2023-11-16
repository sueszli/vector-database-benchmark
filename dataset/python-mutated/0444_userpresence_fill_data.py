from django.db import connection, migrations
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.migrations.state import StateApps

def fill_new_columns(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        while True:
            i = 10
    UserPresence = apps.get_model('zerver', 'UserPresence')
    with connection.cursor() as cursor:
        cursor.execute('SELECT realm_id, user_profile_id, MAX(timestamp) FROM zerver_userpresenceold WHERE status IN (1, 2) GROUP BY realm_id, user_profile_id')
        latest_presence_per_user = cursor.fetchall()
    UserPresence.objects.bulk_create([UserPresence(user_profile_id=presence_row[1], realm_id=presence_row[0], last_connected_time=presence_row[2], last_active_time=presence_row[2]) for presence_row in latest_presence_per_user], batch_size=10000, ignore_conflicts=True)

def clear_new_columns(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        for i in range(10):
            print('nop')
    UserPresence = apps.get_model('zerver', 'UserPresence')
    UserPresence.objects.all().delete()

class Migration(migrations.Migration):
    """
    Ports data from the UserPresence model into the new one.
    """
    atomic = False
    dependencies = [('zerver', '0443_userpresence_new_table_schema')]
    operations = [migrations.RunPython(fill_new_columns, reverse_code=clear_new_columns)]