from django.db import migrations, models

def migrate_db_oracle_version_to_attrs(apps, schema_editor):
    if False:
        while True:
            i = 10
    db_alias = schema_editor.connection.alias
    model = apps.get_model('applications', 'Application')
    oracles = list(model.objects.using(db_alias).filter(type='oracle'))
    for o in oracles:
        o.attrs['version'] = '12c'
    model.objects.using(db_alias).bulk_update(oracles, ['attrs'])

class Migration(migrations.Migration):
    dependencies = [('applications', '0024_alter_application_type')]
    operations = [migrations.RunPython(migrate_db_oracle_version_to_attrs), migrations.AlterUniqueTogether(name='account', unique_together=None), migrations.RemoveField(model_name='account', name='app'), migrations.RemoveField(model_name='account', name='systemuser'), migrations.RemoveField(model_name='application', name='domain'), migrations.RemoveField(model_name='historicalaccount', name='app'), migrations.RemoveField(model_name='historicalaccount', name='history_user'), migrations.RemoveField(model_name='historicalaccount', name='systemuser'), migrations.AlterField(model_name='application', name='category', field=models.CharField(max_length=16, verbose_name='Category')), migrations.AlterField(model_name='application', name='type', field=models.CharField(max_length=16, verbose_name='Type'))]