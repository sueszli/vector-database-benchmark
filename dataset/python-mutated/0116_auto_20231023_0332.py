from django.db import migrations

def migrate_part_responsible_owner(apps, schema_editor):
    if False:
        return 10
    'Copy existing part.responsible field to part.responsible_owner'
    Owner = apps.get_model('users', 'Owner')
    Part = apps.get_model('part', 'Part')
    User = apps.get_model('auth', 'user')
    ContentType = apps.get_model('contenttypes', 'contenttype')
    user_type = ContentType.objects.get_for_model(User)
    parts = Part.objects.exclude(responsible=None)
    for part in parts:
        (owner, _created) = Owner.objects.get_or_create(owner_type=user_type, owner_id=part.responsible.id)
        part.responsible_owner = owner
        part.save()
    if parts.count() > 0:
        print(f"Added 'responsible_owner' for {parts.count()} parts")

def reverse_owner_migration(apps, schema_editor):
    if False:
        while True:
            i = 10
    "Reverse the owner migration:\n\n    - Set the 'responsible' field to a selected user\n    - Only where 'responsible_owner' is set\n    - Only where 'responsible_owner' is a User object\n    "
    Part = apps.get_model('part', 'Part')
    User = apps.get_model('auth', 'user')
    ContentType = apps.get_model('contenttypes', 'contenttype')
    user_type = ContentType.objects.get_for_model(User)
    parts = Part.objects.exclude(responsible_owner=None)
    for part in parts:
        if part.responsible_owner.owner_type == user_type:
            try:
                user = User.objects.get(pk=part.responsible_owner.owner_id)
                part.responsible = user
                part.save()
            except User.DoesNotExist:
                print('User does not exist:', part.responsible_owner.owner_id)
    if parts.count() > 0:
        print(f"Added 'responsible' for {parts.count()} parts")

class Migration(migrations.Migration):
    dependencies = [('part', '0115_part_responsible_owner'), ('users', '0005_owner_model')]
    operations = [migrations.RunPython(migrate_part_responsible_owner, reverse_code=reverse_owner_migration)]