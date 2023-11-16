import time
from django.db import connection, migrations
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.migrations.state import StateApps
from django.db.models import Min
from psycopg2.sql import SQL
BATCH_SIZE = 10000

def sql_copy_id_to_bigint_id(id_range_lower_bound: int, id_range_upper_bound: int) -> None:
    if False:
        i = 10
        return i + 15
    query = SQL('\n            UPDATE zerver_usermessage\n            SET bigint_id = id\n            WHERE id BETWEEN %(lower_bound)s AND %(upper_bound)s\n    ')
    with connection.cursor() as cursor:
        cursor.execute(query, {'lower_bound': id_range_lower_bound, 'upper_bound': id_range_upper_bound})

def copy_id_to_bigid(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        print('Hello World!')
    UserMessage = apps.get_model('zerver', 'UserMessage')
    if not UserMessage.objects.exists():
        return
    first_uncopied_id = UserMessage.objects.filter(bigint_id__isnull=True).aggregate(Min('id'))['id__min']
    last_id = UserMessage.objects.latest('id').id
    id_range_lower_bound = first_uncopied_id
    id_range_upper_bound = first_uncopied_id + BATCH_SIZE
    while id_range_upper_bound <= last_id:
        sql_copy_id_to_bigint_id(id_range_lower_bound, id_range_upper_bound)
        id_range_lower_bound = id_range_upper_bound + 1
        id_range_upper_bound = id_range_lower_bound + BATCH_SIZE
        time.sleep(0.1)
    if last_id > id_range_lower_bound:
        sql_copy_id_to_bigint_id(id_range_lower_bound, last_id)

class Migration(migrations.Migration):
    atomic = False
    dependencies = [('zerver', '0238_usermessage_bigint_id')]
    operations = [migrations.RunSQL("\n        CREATE FUNCTION zerver_usermessage_bigint_id_to_id_trigger_function()\n        RETURNS trigger AS $$\n        BEGIN\n            NEW.bigint_id = NEW.id;\n            RETURN NEW;\n        END\n        $$ LANGUAGE 'plpgsql';\n\n        CREATE TRIGGER zerver_usermessage_bigint_id_to_id_trigger\n        BEFORE INSERT ON zerver_usermessage\n        FOR EACH ROW\n        EXECUTE PROCEDURE zerver_usermessage_bigint_id_to_id_trigger_function();\n        "), migrations.RunPython(copy_id_to_bigid, elidable=True), migrations.RunSQL('\n        CREATE UNIQUE INDEX CONCURRENTLY zerver_usermessage_bigint_id_idx ON zerver_usermessage (bigint_id);\n        ')]