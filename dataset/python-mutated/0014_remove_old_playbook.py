from django.db import migrations

def migrate(apps, schema_editor):
    if False:
        print('Hello World!')
    PlaybookConfig = apps.get_model('playbooks_manager', 'PlaybookConfig')
    PlaybookConfig.objects.get(name='Sample Static Analysis').delete()

def reverse_migrate(apps, schema_editor):
    if False:
        while True:
            i = 10
    PlaybookConfig = apps.get_model('playbooks_manager', 'PlaybookConfig')
    pc = PlaybookConfig.objects.create(type=['file'], name='Sample Static Analysis', description='Execute a static analysis')
    pc.save()
    pc.analyzers.set(['Rtf_Info', 'APKiD', 'Doc_Info', 'ClamAV', 'Cymru_Hash_Registry_Get_File', 'OneNote_Info', 'MalwareBazaar_Get_File', 'YARAify_File_Search', 'PDF_Info', 'BoxJS', 'HybridAnalysis_Get_File', 'Yara', 'OTX_Check_Hash', 'Quark_Engine'])

class Migration(migrations.Migration):
    dependencies = [('playbooks_manager', '0013_alter_playbookconfig_options')]
    operations = [migrations.RunPython(migrate, reverse_migrate)]