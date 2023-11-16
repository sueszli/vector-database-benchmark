from django.db import migrations
from rbac.builtin import BuiltinRole

def create_builtin_roles(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    BuiltinRole.sync_to_db(show_msg=True)

class Migration(migrations.Migration):
    dependencies = [('rbac', '0002_auto_20210929_1409')]
    operations = [migrations.RunPython(create_builtin_roles)]