import os
import sys
import django
import time
app_path = '***** Change me *******'
sys.path.insert(0, app_path)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'jumpserver.settings')
django.setup()
from django.apps import apps
from django.db import connection
from django.db import migrations
from rbac.builtin import BuiltinRole

def migrate_system_role_binding(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    db_alias = schema_editor.connection.alias
    user_model = apps.get_model('users', 'User')
    role_binding_model = apps.get_model('rbac', 'SystemRoleBinding')
    count = 0
    bulk_size = 1000
    while True:
        users = user_model.objects.using(db_alias).only('role', 'id').all()[count:count + bulk_size]
        if not users:
            break
        role_bindings = []
        start = time.time()
        for user in users:
            role = BuiltinRole.get_system_role_by_old_name(user.role)
            role_binding = role_binding_model(scope='system', user_id=user.id, role_id=role.id)
            role_bindings.append(role_binding)
        role_binding_model.objects.bulk_create(role_bindings, ignore_conflicts=True)
        print('\tCreate role binding: {}-{} using: {:.2f}s'.format(count, count + len(users), time.time() - start))
        count += len(users)

class Migration(migrations.Migration):
    dependencies = [('rbac', '0003_auto_20211130_1037')]
    operations = [migrations.RunPython(migrate_system_role_binding)]

def main():
    if False:
        return 10
    schema_editor = connection.schema_editor()
    migrate_system_role_binding(apps, schema_editor)