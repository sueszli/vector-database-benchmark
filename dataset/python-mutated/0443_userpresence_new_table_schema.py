from typing import Any, Callable
import django.db.models.deletion
import django.utils.timezone
from django.conf import settings
from django.db import connection, migrations, models
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.migrations.state import StateApps
from psycopg2.sql import SQL, Identifier

def rename_indexes_constraints(old_table: str, new_table: str) -> Callable[[StateApps, BaseDatabaseSchemaEditor], None]:
    if False:
        print('Hello World!')

    def inner_migration(apps: StateApps, schema_editor: Any) -> None:
        if False:
            print('Hello World!')
        seen_indexes = set()
        with connection.cursor() as cursor:
            constraints = connection.introspection.get_constraints(cursor, old_table)
            for (old_name, infodict) in constraints.items():
                if infodict['check']:
                    suffix = '_check'
                    is_index = False
                elif infodict['foreign_key'] is not None:
                    is_index = False
                    (to_table, to_column) = infodict['foreign_key']
                    suffix = f'_fk_{to_table}_{to_column}'
                elif infodict['primary_key']:
                    suffix = '_pk'
                    is_index = True
                elif infodict['unique']:
                    suffix = '_uniq'
                    is_index = True
                else:
                    suffix = '_idx' if len(infodict['columns']) > 1 else ''
                    is_index = True
                new_name = schema_editor._create_index_name(new_table, infodict['columns'], suffix)
                if new_name in seen_indexes:
                    if is_index:
                        raw_query = SQL('DROP INDEX {old_name}').format(old_name=Identifier(old_name))
                    else:
                        raw_query = SQL('ALTER TABLE {table_name} DROP CONSTRAINT {old_name}').format(table_name=Identifier(old_table), old_name=Identifier(old_name))
                    cursor.execute(raw_query)
                    continue
                seen_indexes.add(new_name)
                if is_index:
                    raw_query = SQL('ALTER INDEX {old_name} RENAME TO {new_name}').format(old_name=Identifier(old_name), new_name=Identifier(new_name))
                else:
                    raw_query = SQL('ALTER TABLE {old_table} RENAME CONSTRAINT {old_name} TO {new_name}').format(old_table=Identifier(old_table), old_name=Identifier(old_name), new_name=Identifier(new_name))
                cursor.execute(raw_query)
            for infodict in connection.introspection.get_sequences(cursor, old_table):
                old_name = infodict['name']
                column = infodict['column']
                new_name = f'{new_table}_{column}_seq'
                raw_query = SQL('ALTER SEQUENCE {old_name} RENAME TO {new_name}').format(old_name=Identifier(old_name), new_name=Identifier(new_name))
                cursor.execute(raw_query)
            cursor.execute(SQL('ALTER TABLE {old_table} RENAME TO {new_table}').format(old_table=Identifier(old_table), new_table=Identifier(new_table)))
    return inner_migration

class Migration(migrations.Migration):
    """
    First step of migrating to a new UserPresence data model. Creates a new
    table with the intended fields, into which in the next step
    data can be ported over from the current UserPresence model.
    In the last step, the old model will be replaced with the new one.
    """
    dependencies = [('zerver', '0442_remove_realmfilter_url_format_string')]
    operations = [migrations.SeparateDatabaseAndState(database_operations=[migrations.RunPython(rename_indexes_constraints('zerver_userpresence', 'zerver_userpresenceold'), reverse_code=rename_indexes_constraints('zerver_userpresenceold', 'zerver_userpresence'))], state_operations=[migrations.RenameModel(old_name='UserPresence', new_name='UserPresenceOld'), migrations.RenameIndex(model_name='userpresenceold', old_name='zerver_userpresence_realm_id_timestamp_25f410da_idx', new_name='zerver_userpresenceold_realm_id_timestamp_52ef5fd3_idx')]), migrations.CreateModel(name='UserPresence', fields=[('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')), ('last_connected_time', models.DateTimeField(db_index=True, default=django.utils.timezone.now, null=True)), ('last_active_time', models.DateTimeField(db_index=True, default=django.utils.timezone.now, null=True)), ('realm', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='zerver.Realm')), ('user_profile', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL))]), migrations.AddIndex(model_name='userpresence', index=models.Index(fields=['realm', 'last_active_time'], name='zerver_userpresence_realm_id_last_active_time_1c5aa9a2_idx')), migrations.AddIndex(model_name='userpresence', index=models.Index(fields=['realm', 'last_connected_time'], name='zerver_userpresence_realm_id_last_connected_time_98d2fc9f_idx'))]