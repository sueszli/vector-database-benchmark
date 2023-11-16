from django.core.management import call_command
from django.db import migrations

class Migration(migrations.Migration):
    dependencies = [('dojo', '0109_group_user_role')]

    def migrate_users(apps, schema_editor):
        if False:
            return 10
        call_command('migrate_authorization_v2')
    operations = [migrations.RunPython(migrate_users)]