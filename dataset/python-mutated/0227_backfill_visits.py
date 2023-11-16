from django.db import migrations
from django.utils import timezone
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

def backfill_visits(apps, schema_editor):
    if False:
        i = 10
        return i + 15
    DiscoverSavedQuery = apps.get_model('sentry', 'DiscoverSavedQuery')
    Dashboard = apps.get_model('sentry', 'Dashboard')
    for saved_query in RangeQuerySetWrapperWithProgressBar(DiscoverSavedQuery.objects.all()):
        changed = False
        if saved_query.visits is None:
            saved_query.visits = 1
            changed = True
        if saved_query.last_visited is None:
            saved_query.last_visited = timezone.now()
            changed = True
        if changed:
            saved_query.save()
    for dashboard in RangeQuerySetWrapperWithProgressBar(Dashboard.objects.all()):
        changed = False
        if dashboard.visits is None:
            dashboard.visits = 1
            changed = True
        if dashboard.last_visited is None:
            dashboard.last_visited = timezone.now()
            changed = True
        if changed:
            dashboard.save()

class Migration(migrations.Migration):
    is_dangerous = False
    atomic = False
    dependencies = [('sentry', '0226_add_visits')]
    operations = [migrations.RunPython(backfill_visits, migrations.RunPython.noop, hints={'tables': ['sentry_discoversavedquery', 'sentry_dashboard']})]