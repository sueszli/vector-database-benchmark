import re
from django.db import migrations

class Migration(migrations.Migration):

    def migrate_data(apps, schema_editor):
        if False:
            i = 10
            return i + 15
        invalid_chars_re = re.compile('[^-._a-zA-Z0-9]')
        ProjectRelationship = apps.get_model('projects', 'ProjectRelationship')
        for p in ProjectRelationship.objects.all():
            if p.alias and invalid_chars_re.match(p.alias):
                new_alias = invalid_chars_re.sub('', p.alias)
                p.alias = new_alias
                p.save()

    def reverse(apps, schema_editor):
        if False:
            print('Hello World!')
        pass
    dependencies = [('projects', '0022_add-alias-slug')]
    operations = [migrations.RunPython(migrate_data, reverse)]