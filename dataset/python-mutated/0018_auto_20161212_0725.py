from django.db import migrations
from django.utils.text import slugify

def create_slugs(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    Value = apps.get_model('product', 'AttributeChoiceValue')
    for value in Value.objects.all():
        value.slug = slugify(value.display)
        value.save()

class Migration(migrations.Migration):
    dependencies = [('product', '0017_attributechoicevalue_slug')]
    operations = [migrations.RunPython(create_slugs, migrations.RunPython.noop)]