from django.db import migrations, models
import sentry.db.models.fields.hybrid_cloud_foreign_key
from sentry.new_migrations.migrations import CheckedMigration

def alertruletriggeraction_migrations():
    if False:
        print('Hello World!')
    database_operations = [migrations.AlterField(model_name='alertruletriggeraction', name='integration', field=sentry.db.models.fields.foreignkey.FlexibleForeignKey('sentry.Integration', db_constraint=False, blank=True, db_index=True, null=True, on_delete=models.CASCADE)), migrations.AlterField(model_name='alertruletriggeraction', name='sentry_app', field=sentry.db.models.fields.foreignkey.FlexibleForeignKey(to='sentry.SentryApp', db_constraint=False, blank=True, db_index=True, null=True))]
    state_operations = [migrations.AlterField(model_name='alertruletriggeraction', name='integration', field=sentry.db.models.fields.hybrid_cloud_foreign_key.HybridCloudForeignKey('sentry.Integration', blank=True, db_index=True, null=True, on_delete='CASCADE')), migrations.RenameField(model_name='alertruletriggeraction', old_name='integration', new_name='integration_id'), migrations.AlterField(model_name='alertruletriggeraction', name='sentry_app', field=sentry.db.models.fields.hybrid_cloud_foreign_key.HybridCloudForeignKey('sentry.SentryApp', blank=True, db_index=True, null=True, on_delete='CASCADE')), migrations.RenameField(model_name='alertruletriggeraction', old_name='sentry_app', new_name='sentry_app_id')]
    return database_operations + [migrations.SeparateDatabaseAndState(state_operations=state_operations)]

class Migration(CheckedMigration):
    is_dangerous = False
    dependencies = [('sentry', '0429_fix_broken_external_issues')]
    operations = alertruletriggeraction_migrations()