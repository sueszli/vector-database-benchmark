from django.db import migrations

def add_cluster_owner(apps, schema_editor):
    if False:
        print('Hello World!')
    Person = apps.get_model('api', 'Person')
    for person in Person.objects.all():
        if person.faces.first():
            person.cluster_owner = person.faces.first().photo.owner
            person.save()

def remove_cluster_owner(apps, schema_editor):
    if False:
        return 10
    Person = apps.get_model('api', 'Person')
    for person in Person.objects.all():
        person.cluster_owner = None

class Migration(migrations.Migration):
    dependencies = [('api', '0031_remove_account')]
    operations = [migrations.RunPython(add_cluster_owner, remove_cluster_owner)]