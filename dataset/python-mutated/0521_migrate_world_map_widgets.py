from django.db import migrations
from sentry.new_migrations.migrations import CheckedMigration

def migrate_worldmap_widgets_to_table_widgets(apps, schema_editor):
    if False:
        print('Hello World!')
    DashboardWidgetQuery = apps.get_model('sentry', 'DashboardWidgetQuery')
    widgetQueries = DashboardWidgetQuery.objects.select_related('widget').filter(widget__display_type=5)
    for widgetQuery in widgetQueries:
        widgetQuery.widget.display_type = 4
        if 'has:geo.country_code' not in widgetQuery.conditions:
            widgetQuery.conditions = widgetQuery.conditions + ' has:geo.country_code'
        if not widgetQuery.columns:
            widgetQuery.columns = []
        if 'geo.region' not in widgetQuery.columns:
            widgetQuery.columns.insert(0, 'geo.region')
        if 'geo.country_code' not in widgetQuery.columns:
            widgetQuery.columns.insert(0, 'geo.country_code')
        if not widgetQuery.fields:
            widgetQuery.fields = []
        if 'geo.region' not in widgetQuery.fields:
            widgetQuery.fields.insert(0, 'geo.region')
        if 'geo.country_code' not in widgetQuery.fields:
            widgetQuery.fields.insert(0, 'geo.country_code')
        widgetQuery.widget.save()
        widgetQuery.save()

class Migration(CheckedMigration):
    is_dangerous = True
    dependencies = [('sentry', '0520_add_flat_file_index_table')]
    operations = [migrations.RunPython(migrate_worldmap_widgets_to_table_widgets, migrations.RunPython.noop, hints={'tables': ['sentry_dashboardwidgetquery', 'sentry_dashboardwidget']})]