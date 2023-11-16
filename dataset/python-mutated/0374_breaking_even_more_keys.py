from django.db import migrations, models
import sentry.db.models.fields.hybrid_cloud_foreign_key
from sentry.new_migrations.migrations import CheckedMigration

def auditlog_organization_migrations():
    if False:
        i = 10
        return i + 15
    database_operations = [migrations.AlterField(model_name='auditlogentry', name='organization', field=sentry.db.models.fields.foreignkey.FlexibleForeignKey(to='sentry.Organization', db_constraint=False, db_index=True, null=False))]
    state_operations = [migrations.AlterField(model_name='auditlogentry', name='organization', field=sentry.db.models.fields.hybrid_cloud_foreign_key.HybridCloudForeignKey('sentry.Organization', db_index=True, on_delete='CASCADE')), migrations.RemoveIndex(model_name='auditlogentry', name='sentry_audi_organiz_588b1e_idx'), migrations.RemoveIndex(model_name='auditlogentry', name='sentry_audi_organiz_c8bd18_idx'), migrations.RenameField(model_name='auditlogentry', old_name='organization', new_name='organization_id'), migrations.AddIndex(model_name='auditlogentry', index=models.Index(fields=['organization_id', 'datetime'], name='sentry_audi_organiz_c8bd18_idx')), migrations.AddIndex(model_name='auditlogentry', index=models.Index(fields=['organization_id', 'event', 'datetime'], name='sentry_audi_organiz_588b1e_idx'))]
    return database_operations + [migrations.SeparateDatabaseAndState(state_operations=state_operations)]

def activityuser_migrations():
    if False:
        while True:
            i = 10
    database_operations = [migrations.AlterField(model_name='activity', name='user', field=sentry.db.models.fields.foreignkey.FlexibleForeignKey(to='sentry.User', db_constraint=False, db_index=True, null=True))]
    state_operations = [migrations.AlterField(model_name='activity', name='user', field=sentry.db.models.fields.hybrid_cloud_foreign_key.HybridCloudForeignKey('sentry.User', db_index=True, on_delete='SET_NULL', null=True)), migrations.RenameField(model_name='activity', old_name='user', new_name='user_id')]
    return database_operations + [migrations.SeparateDatabaseAndState(state_operations=state_operations)]

def recentsearch_user_migrations():
    if False:
        return 10
    database_operations = [migrations.AlterField(model_name='recentsearch', name='user', field=sentry.db.models.fields.foreignkey.FlexibleForeignKey(to='sentry.User', db_constraint=False, db_index=False, null=False))]
    state_operations = [migrations.AlterField(model_name='recentsearch', name='user', field=sentry.db.models.fields.hybrid_cloud_foreign_key.HybridCloudForeignKey('sentry.User', db_index=False, on_delete='CASCADE')), migrations.RenameField(model_name='recentsearch', old_name='user', new_name='user_id'), migrations.AlterUniqueTogether('recentsearch', (('user_id', 'organization', 'type', 'query_hash'),))]
    return database_operations + [migrations.SeparateDatabaseAndState(state_operations=state_operations)]

def dashboard_user_migrations():
    if False:
        for i in range(10):
            print('nop')
    database_operations = [migrations.AlterField(model_name='dashboard', name='created_by', field=sentry.db.models.fields.foreignkey.FlexibleForeignKey(to='sentry.User', db_constraint=False, db_index=True))]
    state_operations = [migrations.AlterField(model_name='dashboard', name='created_by', field=sentry.db.models.fields.hybrid_cloud_foreign_key.HybridCloudForeignKey('sentry.User', on_delete='CASCADE')), migrations.RenameField(model_name='dashboard', old_name='created_by', new_name='created_by_id')]
    return database_operations + [migrations.SeparateDatabaseAndState(state_operations=state_operations)]

class Migration(CheckedMigration):
    is_dangerous = False
    dependencies = [('sentry', '0373_dist_id_to_name')]
    operations = auditlog_organization_migrations() + activityuser_migrations() + recentsearch_user_migrations() + dashboard_user_migrations()