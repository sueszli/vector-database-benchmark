from django.db import migrations
import sentry.db.models.fields.hybrid_cloud_foreign_key
from sentry.new_migrations.migrations import CheckedMigration

def external_actor_and_issue_migrations():
    if False:
        return 10
    database_operations = [migrations.AlterField(model_name='externalissue', name='integration', field=sentry.db.models.fields.foreignkey.FlexibleForeignKey(to='sentry.Integration', db_constraint=False, db_index=True, null=False)), migrations.AlterField(model_name='externalactor', name='integration', field=sentry.db.models.fields.foreignkey.FlexibleForeignKey(to='sentry.Integration', db_constraint=False, db_index=True, null=False))]
    state_operations = [migrations.AlterField(model_name='externalissue', name='integration', field=sentry.db.models.fields.hybrid_cloud_foreign_key.HybridCloudForeignKey('sentry.Integration', db_index=True, on_delete='CASCADE', null=False)), migrations.RenameField(model_name='externalissue', old_name='integration', new_name='integration_id'), migrations.AlterField(model_name='externalactor', name='integration', field=sentry.db.models.fields.hybrid_cloud_foreign_key.HybridCloudForeignKey('sentry.Integration', db_index=True, on_delete='CASCADE', null=False)), migrations.RenameField(model_name='externalactor', old_name='integration', new_name='integration_id'), migrations.AlterUniqueTogether(name='externalissue', unique_together={('organization', 'integration_id', 'key')})]
    return database_operations + [migrations.SeparateDatabaseAndState(state_operations=state_operations)]

class Migration(CheckedMigration):
    is_dangerous = False
    dependencies = [('sentry', '0435_add_alert_rule_source')]
    operations = external_actor_and_issue_migrations()