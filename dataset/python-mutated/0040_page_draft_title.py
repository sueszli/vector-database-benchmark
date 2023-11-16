from django.db import migrations, models
from django.db.models import F

def draft_title(apps, schema_editor):
    if False:
        print('Hello World!')
    Page = apps.get_model('wagtailcore', 'Page')
    Page.objects.all().update(draft_title=F('title'))

class Migration(migrations.Migration):
    dependencies = [('wagtailcore', '0039_collectionviewrestriction')]
    operations = [migrations.AddField(model_name='page', name='draft_title', field=models.CharField(default='', editable=False, max_length=255), preserve_default=False), migrations.RunPython(draft_title, migrations.RunPython.noop)]