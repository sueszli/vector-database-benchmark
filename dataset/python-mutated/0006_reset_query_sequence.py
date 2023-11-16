from django.db import migrations
from django.core.management.color import no_style

def reset_search_promotion_sequence(apps, schema_editor):
    if False:
        print('Hello World!')
    '\n    We set an explicit pk instead of relying on auto-incrementation in migration 0004,\n    so we need to reset the database sequence.\n    '
    Query = apps.get_model('wagtailsearchpromotions.Query')
    QueryDailyHits = apps.get_model('wagtailsearchpromotions.QueryDailyHits')
    statements = schema_editor.connection.ops.sequence_reset_sql(no_style(), [Query, QueryDailyHits])
    for statement in statements:
        schema_editor.execute(statement)

class Migration(migrations.Migration):
    dependencies = [('wagtailsearchpromotions', '0005_switch_query_model')]
    operations = [migrations.RunPython(reset_search_promotion_sequence, migrations.RunPython.noop)]