from django.db import migrations, models

def migrate_platform_type_to_lower(apps, *args):
    if False:
        while True:
            i = 10
    platform_model = apps.get_model('assets', 'Platform')
    platforms = platform_model.objects.all()
    for p in platforms:
        p.type = p.type.lower()
        p.save()

class Migration(migrations.Migration):
    dependencies = [('assets', '0094_auto_20220402_1736')]
    operations = [migrations.RenameField(model_name='platform', old_name='base', new_name='type'), migrations.AddField(model_name='platform', name='category', field=models.CharField(default='host', max_length=32, verbose_name='Category')), migrations.AlterField(model_name='platform', name='type', field=models.CharField(default='linux', max_length=32, verbose_name='Type')), migrations.AddField(model_name='platform', name='domain_enabled', field=models.BooleanField(default=True, verbose_name='Domain enabled')), migrations.AddField(model_name='platform', name='su_enabled', field=models.BooleanField(default=False, verbose_name='Su enabled')), migrations.AddField(model_name='platform', name='su_method', field=models.CharField(blank=True, max_length=32, null=True, verbose_name='Su method')), migrations.RunPython(migrate_platform_type_to_lower)]