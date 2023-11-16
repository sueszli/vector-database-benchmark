from django.db import migrations
from django.db.models import F

def forwards_func(apps, schema_editor):
    if False:
        i = 10
        return i + 15
    Page = apps.get_model('wagtailcore', 'Page')
    Page.objects.filter(has_unpublished_changes=False).update(last_published_at=F('latest_revision_created_at'))

class Migration(migrations.Migration):
    dependencies = [('wagtailcore', '0035_page_last_published_at')]
    operations = [migrations.RunPython(forwards_func, migrations.RunPython.noop)]