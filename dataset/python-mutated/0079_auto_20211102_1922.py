from django.db import migrations

def create_internal_platform(apps, schema_editor):
    if False:
        print('Hello World!')
    model = apps.get_model('assets', 'Platform')
    db_alias = schema_editor.connection.alias
    type_platforms = (('Windows-RDP', 'Windows', {'security': 'rdp'}), ('Windows-TLS', 'Windows', {'security': 'tls'}))
    for (name, base, meta) in type_platforms:
        defaults = {'name': name, 'base': base, 'meta': meta, 'internal': True}
        model.objects.using(db_alias).update_or_create(name=name, defaults=defaults)
    win2016 = model.objects.filter(name='Windows2016').first()
    if win2016:
        win2016.internal = False
        win2016.save(update_fields=['internal'])

class Migration(migrations.Migration):
    dependencies = [('assets', '0078_auto_20211014_2209')]
    operations = [migrations.RunPython(create_internal_platform)]