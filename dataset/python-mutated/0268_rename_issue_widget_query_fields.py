from django.db import migrations
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

def rename_issue_widget_query_fields(apps, schema_editor):
    if False:
        print('Hello World!')
    DashboardWidgetQuery = apps.get_model('sentry', 'DashboardWidgetQuery')
    old_to_new_field_mapping = {'count': 'events', 'userCount': 'users', 'lifetimeCount': 'lifetimeEvents', 'lifetimeUserCount': 'lifetimeUsers'}
    for query in RangeQuerySetWrapperWithProgressBar(DashboardWidgetQuery.objects.all()):
        fields = query.fields
        new_fields = [old_to_new_field_mapping.get(field, field) for field in fields]
        if fields != new_fields:
            query.fields = new_fields
            query.save()

class Migration(migrations.Migration):
    is_dangerous = False
    atomic = False
    dependencies = [('sentry', '0267_sentry_release_version_btree')]
    operations = [migrations.RunPython(rename_issue_widget_query_fields, migrations.RunPython.noop, hints={'tables': ['sentry_dashboardwidgetquery']})]