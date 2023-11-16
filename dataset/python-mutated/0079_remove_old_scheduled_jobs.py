from django.db import migrations
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.migrations.state import StateApps

def delete_old_scheduled_jobs(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        i = 10
        return i + 15
    "Delete any old scheduled jobs, to handle changes in the format of\n    that table.  Ideally, we'd translate the jobs, but it's not really\n    worth the development effort to save a few invitation reminders\n    and day2 followup emails.\n    "
    ScheduledJob = apps.get_model('zerver', 'ScheduledJob')
    ScheduledJob.objects.all().delete()

class Migration(migrations.Migration):
    dependencies = [('zerver', '0078_service')]
    operations = [migrations.RunPython(delete_old_scheduled_jobs, elidable=True)]