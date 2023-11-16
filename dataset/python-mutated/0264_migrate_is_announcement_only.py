from django.db import migrations
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.migrations.state import StateApps

def upgrade_stream_post_policy(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        return 10
    Stream = apps.get_model('zerver', 'Stream')
    Stream.STREAM_POST_POLICY_EVERYONE = 1
    Stream.STREAM_POST_POLICY_ADMINS = 2
    Stream.objects.filter(is_announcement_only=False).update(stream_post_policy=Stream.STREAM_POST_POLICY_EVERYONE)
    Stream.objects.filter(is_announcement_only=True).update(stream_post_policy=Stream.STREAM_POST_POLICY_ADMINS)

class Migration(migrations.Migration):
    dependencies = [('zerver', '0263_stream_stream_post_policy')]
    operations = [migrations.RunPython(upgrade_stream_post_policy, reverse_code=migrations.RunPython.noop, elidable=True)]