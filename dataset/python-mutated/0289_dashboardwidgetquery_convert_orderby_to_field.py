import re
from django.db import migrations
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar
AGGREGATE_PATTERN = '^(\\w+)\\((.*)?\\)$'

def get_function_alias(field: str):
    if False:
        for i in range(10):
            print('nop')
    match = re.match(AGGREGATE_PATTERN, field)
    if not match:
        return field
    if not match[2]:
        return match[1]
    arguments = parse_arguments(match[1], match[2])
    alias = match[1] + '_' + '_'.join(arguments)
    for pattern in ['[^\\w]', '^_+', '_+$']:
        alias = re.sub(pattern, '_', alias)
    return alias

def parse_arguments(function: str, columns: str):
    if False:
        return 10
    '\n    Some functions take a quoted string for their arguments that may contain commas,\n    which requires special handling.\n    This function attempts to be identical with the similarly named parse_arguments\n    found in static/app/utils/discover/fields.tsx\n    '
    if function != 'to_other' and function != 'count_if' and (function != 'spans_histogram') or len(columns) == 0:
        return [c.strip() for c in columns.split(',') if len(c.strip()) > 0]
    args = []
    quoted = False
    escaped = False
    (i, j) = (0, 0)
    while j < len(columns):
        if i == j and columns[j] == '"':
            quoted = True
        elif i == j and columns[j] == ' ':
            i += 1
        elif quoted and (not escaped) and (columns[j] == '\\'):
            escaped = True
        elif quoted and (not escaped) and (columns[j] == '"'):
            quoted = False
        elif quoted and escaped:
            escaped = False
        elif quoted and columns[j] == ',':
            pass
        elif columns[j] == ',':
            args.append(columns[i:j].strip())
            i = j + 1
        j += 1
    if i != j:
        args.append(columns[i:].strip())
    return [arg for arg in args if arg]

def convert_dashboard_widget_query_orderby_to_function_format(apps, schema_editor):
    if False:
        print('Hello World!')
    DashboardWidgetQuery = apps.get_model('sentry', 'DashboardWidgetQuery')
    for query in RangeQuerySetWrapperWithProgressBar(DashboardWidgetQuery.objects.all()):
        if query.orderby == '':
            continue
        orderby = query.orderby
        stripped_orderby = orderby.lstrip('-')
        orderby_prefix = '-' if orderby.startswith('-') else ''
        for aggregate in query.aggregates:
            alias = get_function_alias(aggregate)
            if alias == stripped_orderby:
                query.orderby = f'{orderby_prefix}{aggregate}'
                query.save()
                continue

class Migration(CheckedMigration):
    is_dangerous = False
    atomic = False
    dependencies = [('sentry', '0288_fix_savedsearch_state')]
    operations = [migrations.RunPython(convert_dashboard_widget_query_orderby_to_function_format, migrations.RunPython.noop, hints={'tables': ['sentry_dashboardwidgetquery']})]