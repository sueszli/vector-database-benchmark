import django
from django.db import migrations

def set_replication_identity(schema_editor, model, column_names):
    if False:
        while True:
            i = 10
    cursor = schema_editor.connection.cursor()
    unique_constraint_names = schema_editor._constraint_names(model, column_names, unique=True)
    if not unique_constraint_names:
        if django.VERSION[:2] < (2, 0):
            index = schema_editor._create_index_name(model, column_names, '_uniq')
        else:
            index = schema_editor._create_index_name(model._meta.db_table, column_names, '_uniq')
        cursor.execute(f"CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS {schema_editor.quote_name(index)} ON {schema_editor.quote_name(model._meta.db_table)} ({', '.join((schema_editor.quote_name(col) for col in column_names))});")
    else:
        index = unique_constraint_names[0]
    cursor.execute(f'ALTER TABLE {schema_editor.quote_name(model._meta.db_table)} REPLICA IDENTITY USING INDEX {schema_editor.quote_name(index)}')
    cursor.close()

def reset_replication_identity(schema_editor, model):
    if False:
        print('Hello World!')
    cursor = schema_editor.connection.cursor()
    cursor.execute(f'ALTER TABLE {schema_editor.quote_name(model._meta.db_table)} REPLICA IDENTITY DEFAULT')
    cursor.close()

def set_groupassignee_replication_identity(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    group_assignee_model = apps.get_model('sentry', 'GroupAssignee')
    set_replication_identity(schema_editor, group_assignee_model, ['project_id', 'group_id'])

def reset_groupassignee_replication_identity(apps, schema_editor):
    if False:
        return 10
    group_assignee_model = apps.get_model('sentry', 'GroupAssignee')
    reset_replication_identity(schema_editor, group_assignee_model)

def set_groupedmessage_replication_identity(apps, schema_editor):
    if False:
        print('Hello World!')
    group_model = apps.get_model('sentry', 'Group')
    set_replication_identity(schema_editor, group_model, ['project_id', 'id'])

def reset_groupedmessage_replication_identity(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    group_model = apps.get_model('sentry', 'Group')
    reset_replication_identity(schema_editor, group_model)

class Migration(migrations.Migration):
    is_dangerous = True
    atomic = False
    dependencies = [('sentry', '0215_fix_state')]
    operations = [migrations.SeparateDatabaseAndState(database_operations=[migrations.RunPython(code=set_groupassignee_replication_identity, reverse_code=reset_groupassignee_replication_identity, atomic=False, hints={'tables': ['sentry_groupasignee']})], state_operations=[migrations.AlterUniqueTogether(name='groupassignee', unique_together={('project', 'group')})]), migrations.SeparateDatabaseAndState(database_operations=[migrations.RunPython(code=set_groupedmessage_replication_identity, reverse_code=reset_groupedmessage_replication_identity, atomic=False, hints={'tables': ['sentry_groupedmessage']})], state_operations=[migrations.AlterUniqueTogether(name='group', unique_together={('project', 'id'), ('project', 'short_id')})])]