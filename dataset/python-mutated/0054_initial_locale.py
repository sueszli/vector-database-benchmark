from django.conf import settings
from django.db import migrations
from wagtail.coreutils import get_supported_content_language_variant

def initial_locale(apps, schema_editor):
    if False:
        print('Hello World!')
    Locale = apps.get_model('wagtailcore.Locale')
    Locale.objects.create(language_code=get_supported_content_language_variant(settings.LANGUAGE_CODE))

class Migration(migrations.Migration):
    dependencies = [('wagtailcore', '0053_locale_model')]
    operations = [migrations.RunPython(initial_locale, migrations.RunPython.noop)]