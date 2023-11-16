from django.db import migrations
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

def backfill_popularity(apps, schema_editor):
    if False:
        print('Hello World!')
    SentryApp = apps.get_model('sentry', 'SentryApp')
    for sentry_app in RangeQuerySetWrapperWithProgressBar(SentryApp.objects.all()):
        if sentry_app.popularity is None:
            sentry_app.popularity = 1
            sentry_app.save()

class Migration(migrations.Migration):
    is_dangerous = False
    atomic = False
    dependencies = [('sentry', '0249_add_avatar_type_back')]
    operations = [migrations.RunPython(backfill_popularity, migrations.RunPython.noop, hints={'tables': ['sentry_sentryapp']})]