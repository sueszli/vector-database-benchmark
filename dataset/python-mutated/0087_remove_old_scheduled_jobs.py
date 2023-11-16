from django.db import migrations
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.migrations.state import StateApps

def delete_old_scheduled_jobs(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        print('Hello World!')
    "Delete any old scheduled jobs, to handle changes in the format of\n    send_email. Ideally, we'd translate the jobs, but it's not really\n    worth the development effort to save a few invitation reminders\n    and day2 followup emails.\n    "
    ScheduledJob = apps.get_model('zerver', 'ScheduledJob')
    ScheduledJob.objects.all().delete()

class Migration(migrations.Migration):
    dependencies = [('zerver', '0086_realm_alter_default_org_type')]
    operations = [migrations.RunPython(delete_old_scheduled_jobs, elidable=True)]