from django.conf import settings
from django.db import migrations, models
import sentry.db.models.fields.hybrid_cloud_foreign_key
from sentry.new_migrations.migrations import CheckedMigration

def break_inviter_fk():
    if False:
        for i in range(10):
            print('nop')
    database_operations = [migrations.AlterField(model_name='organizationmember', name='inviter', field=sentry.db.models.fields.foreignkey.FlexibleForeignKey(blank=True, null=True, on_delete=models.deletion.CASCADE, related_name='sentry_inviter_set', to=settings.AUTH_USER_MODEL, db_constraint=False, db_index=True))]
    state_operations = [migrations.AlterField(model_name='organizationmember', name='inviter', field=sentry.db.models.fields.hybrid_cloud_foreign_key.HybridCloudForeignKey(settings.AUTH_USER_MODEL, blank=True, db_index=True, null=True, on_delete='SET_NULL')), migrations.RenameField(model_name='organizationmember', old_name='inviter', new_name='inviter_id')]
    return database_operations + [migrations.SeparateDatabaseAndState(state_operations=state_operations)]

class Migration(CheckedMigration):
    is_dangerous = False
    dependencies = [('sentry', '0437_remove_fk_notifications_target')]
    operations = break_inviter_fk()