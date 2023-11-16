from django.db import migrations, models

def update_builtin_org(apps, schema_editor):
    if False:
        print('Hello World!')
    org_model = apps.get_model('orgs', 'Organization')
    org_model.objects.create(id='00000000-0000-0000-0000-000000000004', name='SYSTEM', builtin=True)
    org_model.objects.filter(name='DEFAULT').update(builtin=True)

class Migration(migrations.Migration):
    dependencies = [('orgs', '0013_alter_organization_options')]
    operations = [migrations.AddField(model_name='organization', name='builtin', field=models.BooleanField(default=False, verbose_name='Builtin')), migrations.RunPython(update_builtin_org), migrations.DeleteModel(name='OrganizationMember')]