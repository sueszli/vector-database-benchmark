from django.db import migrations
UNKNOWN_PERSON_NAME = 'Unknown - Other'
KIND_UNKNOWN = 'UNKNOWN'

def migrate_unknown(apps, schema_editor):
    if False:
        return 10
    Person = apps.get_model('api', 'Person')
    person: Person
    try:
        person = Person.objects.get(name='unknown')
        person.name = UNKNOWN_PERSON_NAME
        person.kind = KIND_UNKNOWN
        person.save()
    except Person.DoesNotExist:
        unknown_person: Person = Person.objects.get_or_create(name=UNKNOWN_PERSON_NAME, cluster_owner=None, kind=KIND_UNKNOWN)[0]
        if unknown_person.kind != KIND_UNKNOWN:
            unknown_person.kind = KIND_UNKNOWN
            unknown_person.save()

def unmigrate_unknown(apps, schema_editor):
    if False:
        return 10
    Person = apps.get_model('api', 'Person')
    try:
        person: Person = Person.objects.get(name=UNKNOWN_PERSON_NAME)
        person.name = 'unknown'
        person.kind = ''
        person.save()
    except Person.DoesNotExist:
        pass

class Migration(migrations.Migration):
    dependencies = [('api', '0026_add_cluster_info')]
    operations = [migrations.RunPython(migrate_unknown, unmigrate_unknown)]