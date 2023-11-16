from django.db import migrations

def reset_insight_refreshing_status(apps, _) -> None:
    if False:
        return 10
    pass

def reverse(_apps, _schema_editor) -> None:
    if False:
        for i in range(10):
            print('nop')
    pass

class Migration(migrations.Migration):
    dependencies = [('posthog', '0251_event_buffer')]
    operations = [migrations.RunPython(reset_insight_refreshing_status, reverse, elidable=True)]