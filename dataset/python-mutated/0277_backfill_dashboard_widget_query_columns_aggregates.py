import re
from django.db import migrations
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar
AGGREGATE_PATTERN = '^(\\w+)\\((.*)?\\)$'
AGGREGATE_BASE = '.*(\\w+)\\((.*)?\\)'
EQUATION_PREFIX = 'equation|'

def is_equation(field: str) -> bool:
    if False:
        while True:
            i = 10
    'check if a public alias is an equation, which start with the equation prefix\n    eg. `equation|5 + 5`\n    '
    return field.startswith(EQUATION_PREFIX)

def is_aggregate(field: str) -> bool:
    if False:
        print('Hello World!')
    field_match = re.match(AGGREGATE_PATTERN, field)
    if field_match:
        return True
    equation_match = re.match(AGGREGATE_BASE, field)
    if equation_match and is_equation(field):
        return True
    return False

def backfill_columns_aggregates(apps, schema_editor):
    if False:
        print('Hello World!')
    DashboardWidgetQuery = apps.get_model('sentry', 'DashboardWidgetQuery')
    for widget_query in RangeQuerySetWrapperWithProgressBar(DashboardWidgetQuery.objects.all()):
        if widget_query.columns or widget_query.aggregates:
            continue
        fields = widget_query.fields or []
        columns = []
        aggregates = []
        for field in fields:
            if is_aggregate(field):
                aggregates.append(field)
            else:
                columns.append(field)
        widget_query.columns = columns
        widget_query.aggregates = aggregates
        widget_query.save()

class Migration(CheckedMigration):
    is_dangerous = False
    atomic = False
    dependencies = [('sentry', '0276_rulefirehistory_date_added_index')]
    operations = [migrations.RunPython(backfill_columns_aggregates, migrations.RunPython.noop, hints={'tables': ['sentry_dashboardwidgetquery']})]