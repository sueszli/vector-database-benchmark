from django.db import connection, migrations

def migrate(apps, schema_editor):
    if False:
        return 10
    with connection.cursor() as cursor:
        cursor.execute('CREATE INDEX CONCURRENTLY IF NOT EXISTS posthog_eve_created_6a34ca_idx ON posthog_event(created_at)')

def backwards(apps, schema_editor):
    if False:
        i = 10
        return i + 15
    pass

class Migration(migrations.Migration):
    atomic = False
    dependencies = [('posthog', '0140_team_data_attributes')]
    operations = [migrations.RunPython(migrate, backwards, elidable=True)]