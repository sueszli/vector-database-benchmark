from django.db import migrations, models

def add_default_reference(apps, schema_editor):
    if False:
        i = 10
        return i + 15
    '\n    Add a "default" build-order reference for any existing build orders.\n    Best we can do is use the PK of the build order itself.\n    '
    Build = apps.get_model('build', 'build')
    count = 0
    for build in Build.objects.all():
        build.reference = str(build.pk)
        build.save()
        count += 1
    if count > 0:
        print(f'\nUpdated build reference for {count} existing BuildOrder objects')

class Migration(migrations.Migration):
    atomic = False
    dependencies = [('build', '0017_auto_20200426_0612')]
    operations = [migrations.AddField(model_name='build', name='reference', field=models.CharField(help_text='Build Order Reference', blank=True, max_length=64, unique=False, verbose_name='Reference')), migrations.RunPython(add_default_reference, reverse_code=migrations.RunPython.noop), migrations.AlterField(model_name='build', name='reference', field=models.CharField(help_text='Build Order Reference', max_length=64, blank=False, unique=True, verbose_name='Reference'))]