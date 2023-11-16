import time
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
    print('')
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

def migrate_org_role_binding(apps, schema_editor):
    if False:
        while True:
            i = 10
    db_alias = schema_editor.connection.alias
    org_member_model = apps.get_model('orgs', 'OrganizationMember')
    role_binding_model = apps.get_model('rbac', 'RoleBinding')
    count = 0
    bulk_size = 1000
    while True:
        members = org_member_model.objects.using(db_alias).only('role', 'user_id', 'org_id').all()[count:count + bulk_size]
        if not members:
            break
        role_bindings = []
        start = time.time()
        for member in members:
            role = BuiltinRole.get_org_role_by_old_name(member.role)
            role_binding = role_binding_model(scope='org', user_id=member.user_id, role_id=role.id, org_id=member.org_id)
            role_bindings.append(role_binding)
        role_binding_model.objects.bulk_create(role_bindings, ignore_conflicts=True)
        print('\tCreate role binding: {}-{} using: {:.2f}s'.format(count, count + len(members), time.time() - start))
        count += len(members)

class Migration(migrations.Migration):
    dependencies = [('rbac', '0003_auto_20211130_1037')]
    operations = [migrations.RunPython(migrate_system_role_binding), migrations.RunPython(migrate_org_role_binding)]