from django.db import migrations, models
import sentry.db.models.fields.hybrid_cloud_foreign_key
from sentry.new_migrations.migrations import CheckedMigration

def exported_data_user_migrations():
    if False:
        print('Hello World!')
    database_operations = [migrations.AlterField(model_name='exporteddata', name='user', field=sentry.db.models.fields.foreignkey.FlexibleForeignKey(to='sentry.User', db_constraint=False, db_index=True, null=True))]
    state_operations = [migrations.AlterField(model_name='exporteddata', name='user', field=sentry.db.models.fields.hybrid_cloud_foreign_key.HybridCloudForeignKey('sentry.User', db_index=True, null=True, on_delete='SET_NULL')), migrations.RenameField(model_name='exporteddata', old_name='user', new_name='user_id')]
    return database_operations + [migrations.SeparateDatabaseAndState(state_operations=state_operations)]

def alertruleactivity_user_migrations():
    if False:
        for i in range(10):
            print('nop')
    database_operations = [migrations.AlterField(model_name='alertruleactivity', name='user', field=sentry.db.models.fields.foreignkey.FlexibleForeignKey(to='sentry.User', db_constraint=False, db_index=True, null=True))]
    state_operations = [migrations.AlterField(model_name='alertruleactivity', name='user', field=sentry.db.models.fields.hybrid_cloud_foreign_key.HybridCloudForeignKey('sentry.User', db_index=True, null=True, on_delete='SET_NULL')), migrations.RenameField(model_name='alertruleactivity', old_name='user', new_name='user_id')]
    return database_operations + [migrations.SeparateDatabaseAndState(state_operations=state_operations)]

def discover_saved_query_user_migrations():
    if False:
        i = 10
        return i + 15
    database_operations = [migrations.AlterField(model_name='discoversavedquery', name='created_by', field=sentry.db.models.fields.foreignkey.FlexibleForeignKey(to='sentry.User', db_constraint=False, db_index=True, null=True))]
    state_operations = [migrations.RemoveConstraint(model_name='discoversavedquery', name='unique_user_homepage_query'), migrations.AddConstraint(model_name='discoversavedquery', constraint=models.UniqueConstraint(condition=models.Q(is_homepage=True), fields=('organization', 'created_by_id', 'is_homepage'), name='unique_user_homepage_query')), migrations.AlterField(model_name='discoversavedquery', name='created_by', field=sentry.db.models.fields.hybrid_cloud_foreign_key.HybridCloudForeignKey('sentry.User', db_index=True, null=True, on_delete='SET_NULL')), migrations.RenameField(model_name='discoversavedquery', old_name='created_by', new_name='created_by_id')]
    return database_operations + [migrations.SeparateDatabaseAndState(state_operations=state_operations)]

def incidentsubscription_user_migrations():
    if False:
        return 10
    database_operations = [migrations.AlterField(model_name='incidentsubscription', name='user', field=sentry.db.models.fields.foreignkey.FlexibleForeignKey(to='sentry.User', db_constraint=False, db_index=True))]
    state_operations = [migrations.AlterField(model_name='incidentsubscription', name='user', field=sentry.db.models.fields.hybrid_cloud_foreign_key.HybridCloudForeignKey('sentry.User', db_index=True, on_delete='CASCADE')), migrations.RenameField(model_name='incidentsubscription', old_name='user', new_name='user_id'), migrations.AlterUniqueTogether(name='incidentsubscription', unique_together={('incident', 'user_id')})]
    return database_operations + [migrations.SeparateDatabaseAndState(state_operations=state_operations)]

def incidentactivity_user_migrations():
    if False:
        return 10
    database_operations = [migrations.AlterField(model_name='incidentactivity', name='user', field=sentry.db.models.fields.foreignkey.FlexibleForeignKey(to='sentry.User', db_constraint=False, db_index=True, null=True))]
    state_operations = [migrations.AlterField(model_name='incidentactivity', name='user', field=sentry.db.models.fields.hybrid_cloud_foreign_key.HybridCloudForeignKey('sentry.User', db_index=True, on_delete='CASCADE', null=True)), migrations.RenameField(model_name='incidentactivity', old_name='user', new_name='user_id')]
    return database_operations + [migrations.SeparateDatabaseAndState(state_operations=state_operations)]

def incidentseen_user_migrations():
    if False:
        for i in range(10):
            print('nop')
    database_operations = [migrations.AlterField(model_name='incidentseen', name='user', field=sentry.db.models.fields.foreignkey.FlexibleForeignKey(to='sentry.User', db_constraint=False, db_index=False))]
    state_operations = [migrations.AlterField(model_name='incidentseen', name='user', field=sentry.db.models.fields.hybrid_cloud_foreign_key.HybridCloudForeignKey('sentry.User', db_index=False, on_delete='CASCADE')), migrations.RenameField(model_name='incidentseen', old_name='user', new_name='user_id'), migrations.AlterUniqueTogether(name='incidentseen', unique_together={('user_id', 'incident')})]
    return database_operations + [migrations.SeparateDatabaseAndState(state_operations=state_operations)]

def projecttransactionthresholdoverride_user_migrations():
    if False:
        print('Hello World!')
    database_operations = [migrations.AlterField(model_name='projecttransactionthreshold', name='edited_by', field=sentry.db.models.fields.foreignkey.FlexibleForeignKey(to='sentry.User', db_constraint=False, db_index=True, null=True))]
    state_operations = [migrations.AlterField(model_name='projecttransactionthreshold', name='edited_by', field=sentry.db.models.fields.hybrid_cloud_foreign_key.HybridCloudForeignKey('sentry.User', null=True, on_delete='SET_NULL')), migrations.RenameField(model_name='projecttransactionthreshold', old_name='edited_by', new_name='edited_by_id')]
    return database_operations + [migrations.SeparateDatabaseAndState(state_operations=state_operations)]

def project_transaction_threshold_user_migrations():
    if False:
        i = 10
        return i + 15
    database_operations = [migrations.AlterField(model_name='projecttransactionthresholdoverride', name='edited_by', field=sentry.db.models.fields.foreignkey.FlexibleForeignKey(to='sentry.User', db_constraint=False, db_index=True, null=True))]
    state_operations = [migrations.AlterField(model_name='projecttransactionthresholdoverride', name='edited_by', field=sentry.db.models.fields.hybrid_cloud_foreign_key.HybridCloudForeignKey('sentry.User', null=True, on_delete='SET_NULL')), migrations.RenameField(model_name='projecttransactionthresholdoverride', old_name='edited_by', new_name='edited_by_id')]
    return database_operations + [migrations.SeparateDatabaseAndState(state_operations=state_operations)]

class Migration(CheckedMigration):
    is_dangerous = False
    dependencies = [('sentry', '0364_remove_project_id_from_environment')]
    operations = exported_data_user_migrations() + discover_saved_query_user_migrations() + alertruleactivity_user_migrations() + incidentsubscription_user_migrations() + incidentactivity_user_migrations() + incidentseen_user_migrations() + project_transaction_threshold_user_migrations() + projecttransactionthresholdoverride_user_migrations()