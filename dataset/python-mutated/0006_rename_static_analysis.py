from django.db import migrations

def migrate(apps, schema_editor):
    if False:
        while True:
            i = 10
    PlaybookConfig = apps.get_model('playbooks_manager', 'PlaybookConfig')
    pc = PlaybookConfig.objects.get(name='Sample Static Analsis')
    pc.name = 'Sample Static Analysis'
    pc.save()
    PlaybookConfig.objects.get(name='Sample Static Analsis').delete()

def reverse_migrate(apps, schema_editor):
    if False:
        print('Hello World!')
    PlaybookConfig = apps.get_model('playbooks_manager', 'PlaybookConfig')
    pc = PlaybookConfig.objects.get(name='Sample Static Analysis')
    pc.name = 'Sample Static Analsis'
    pc.save()
    PlaybookConfig.objects.get(name='Sample Static Analysis').delete()

class Migration(migrations.Migration):
    dependencies = [('playbooks_manager', '0005_static_analysis')]
    operations = [migrations.RunPython(migrate, reverse_migrate)]