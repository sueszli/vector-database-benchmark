from django.db import migrations
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

def delete_removed_plugin_data(apps, schema_editor):
    if False:
        return 10
    "\n    Delete the rows in the ProjectOption table that relate to plugins we've deleted.\n    "
    ProjectOption = apps.get_model('sentry', 'ProjectOption')
    for project_option in RangeQuerySetWrapperWithProgressBar(ProjectOption.objects.all()):
        if project_option.key in ('jira-ac:enabled', 'vsts:default_project', 'vsts:enabled', 'vsts:instance', 'clubhouse:enabled', 'clubhouse:project', 'clubhouse:token', 'teamwork:enabled', 'teamwork:token', 'teamwork:url'):
            project_option.delete()

class Migration(migrations.Migration):
    is_dangerous = True
    atomic = False
    dependencies = [('sentry', '0241_grouphistory_null_actor')]
    operations = [migrations.RunPython(delete_removed_plugin_data, migrations.RunPython.noop, hints={'tables': ['sentry_projectoptions']})]