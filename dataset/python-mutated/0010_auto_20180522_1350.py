from __future__ import unicode_literals
from django.db import migrations

def migrate_org_to_orgs(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    ff_admin_user_model = apps.get_model('users', 'FFAdminUser')
    for user in ff_admin_user_model.objects.all():
        if user.organisation is not None:
            user.add_organisation(user.organisation)

class Migration(migrations.Migration):
    dependencies = [('users', '0009_auto_20180522_1342')]
    operations = [migrations.RunPython(migrate_org_to_orgs)]