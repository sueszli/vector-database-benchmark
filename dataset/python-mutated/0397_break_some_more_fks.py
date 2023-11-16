from django.db import migrations
import sentry.db.models.fields.hybrid_cloud_foreign_key
from sentry.new_migrations.migrations import CheckedMigration

def apikey_migrations():
    if False:
        return 10
    database_operations = [migrations.AlterField(model_name='apikey', name='organization', field=sentry.db.models.fields.foreignkey.FlexibleForeignKey(to='sentry.Organization', db_constraint=False, db_index=True, null=False))]
    state_operations = [migrations.AlterField(model_name='apikey', name='organization', field=sentry.db.models.fields.hybrid_cloud_foreign_key.HybridCloudForeignKey('sentry.Organization', db_index=True, on_delete='CASCADE', null=False)), migrations.RenameField(model_name='apikey', old_name='organization', new_name='organization_id')]
    return database_operations + [migrations.SeparateDatabaseAndState(state_operations=state_operations)]

def authprovider_migrations():
    if False:
        for i in range(10):
            print('nop')
    database_operations = [migrations.AlterField(model_name='authprovider', name='organization', field=sentry.db.models.fields.foreignkey.FlexibleForeignKey(to='sentry.Organization', db_constraint=False, db_index=True, null=False, unique=True))]
    state_operations = [migrations.AlterField(model_name='authprovider', name='organization', field=sentry.db.models.fields.hybrid_cloud_foreign_key.HybridCloudForeignKey('sentry.Organization', db_index=True, on_delete='CASCADE', null=False, unique=True)), migrations.RenameField(model_name='authprovider', old_name='organization', new_name='organization_id')]
    return database_operations + [migrations.SeparateDatabaseAndState(state_operations=state_operations)]

def organizationintegration_migrations():
    if False:
        print('Hello World!')
    database_operations = [migrations.AlterField(model_name='organizationintegration', name='organization', field=sentry.db.models.fields.foreignkey.FlexibleForeignKey(to='sentry.Organization', db_constraint=False, db_index=True, null=False))]
    state_operations = [migrations.AlterField(model_name='organizationintegration', name='organization', field=sentry.db.models.fields.hybrid_cloud_foreign_key.HybridCloudForeignKey('sentry.Organization', db_index=True, on_delete='CASCADE', null=False)), migrations.RenameField(model_name='organizationintegration', old_name='organization', new_name='organization_id'), migrations.AlterUniqueTogether('organizationintegration', (('organization_id', 'integration'),)), migrations.RemoveField(model_name='integration', name='organizations')]
    return database_operations + [migrations.SeparateDatabaseAndState(state_operations=state_operations)]

def sentryapp_migrations():
    if False:
        for i in range(10):
            print('nop')
    database_operations = [migrations.AlterField(model_name='sentryapp', name='owner', field=sentry.db.models.fields.foreignkey.FlexibleForeignKey(to='sentry.Organization', db_constraint=False, db_index=True, null=False))]
    state_operations = [migrations.AlterField(model_name='sentryapp', name='owner', field=sentry.db.models.fields.hybrid_cloud_foreign_key.HybridCloudForeignKey('sentry.Organization', db_index=True, on_delete='CASCADE', null=False)), migrations.RenameField(model_name='sentryapp', old_name='owner', new_name='owner_id')]
    return database_operations + [migrations.SeparateDatabaseAndState(state_operations=state_operations)]

def sentryappinstallation_migrations():
    if False:
        i = 10
        return i + 15
    database_operations = [migrations.AlterField(model_name='sentryappinstallation', name='organization', field=sentry.db.models.fields.foreignkey.FlexibleForeignKey(to='sentry.Organization', db_constraint=False, db_index=True, null=False))]
    state_operations = [migrations.AlterField(model_name='sentryappinstallation', name='organization', field=sentry.db.models.fields.hybrid_cloud_foreign_key.HybridCloudForeignKey('sentry.Organization', db_index=True, on_delete='CASCADE', null=False)), migrations.RenameField(model_name='sentryappinstallation', old_name='organization', new_name='organization_id')]
    return database_operations + [migrations.SeparateDatabaseAndState(state_operations=state_operations)]

def sentryappinstallationforprovider_migrations():
    if False:
        print('Hello World!')
    database_operations = [migrations.AlterField(model_name='sentryappinstallationforprovider', name='organization', field=sentry.db.models.fields.foreignkey.FlexibleForeignKey(to='sentry.Organization', db_constraint=False, db_index=True, null=False))]
    state_operations = [migrations.AlterField(model_name='sentryappinstallationforprovider', name='organization', field=sentry.db.models.fields.hybrid_cloud_foreign_key.HybridCloudForeignKey('sentry.Organization', db_index=True, on_delete='CASCADE', null=False)), migrations.RenameField(model_name='sentryappinstallationforprovider', old_name='organization', new_name='organization_id')]
    return database_operations + [migrations.SeparateDatabaseAndState(state_operations=state_operations)]

def organizationaccessrequest_migrations():
    if False:
        while True:
            i = 10
    database_operations = [migrations.AlterField(model_name='organizationaccessrequest', name='requester', field=sentry.db.models.fields.foreignkey.FlexibleForeignKey(to='sentry.User', db_constraint=False, db_index=True, null=True))]
    state_operations = [migrations.AlterField(model_name='organizationaccessrequest', name='requester', field=sentry.db.models.fields.hybrid_cloud_foreign_key.HybridCloudForeignKey('sentry.User', db_index=True, on_delete='CASCADE', null=True)), migrations.RenameField(model_name='organizationaccessrequest', old_name='requester', new_name='requester_id')]
    return database_operations + [migrations.SeparateDatabaseAndState(state_operations=state_operations)]

def organizationonboardingtask_migrations():
    if False:
        print('Hello World!')
    database_operations = [migrations.AlterField(model_name='organizationonboardingtask', name='user', field=sentry.db.models.fields.foreignkey.FlexibleForeignKey(to='sentry.User', db_constraint=False, db_index=True, null=True))]
    state_operations = [migrations.AlterField(model_name='organizationonboardingtask', name='user', field=sentry.db.models.fields.hybrid_cloud_foreign_key.HybridCloudForeignKey('sentry.User', db_index=True, on_delete='SET_NULL', null=True)), migrations.RenameField(model_name='organizationonboardingtask', old_name='user', new_name='user_id')]
    return database_operations + [migrations.SeparateDatabaseAndState(state_operations=state_operations)]

def projectbookmark_migrations():
    if False:
        i = 10
        return i + 15
    database_operations = [migrations.AlterField(model_name='projectbookmark', name='user', field=sentry.db.models.fields.foreignkey.FlexibleForeignKey(to='sentry.User', db_constraint=False, db_index=True, null=False))]
    state_operations = [migrations.AlterField(model_name='projectbookmark', name='user', field=sentry.db.models.fields.hybrid_cloud_foreign_key.HybridCloudForeignKey('sentry.User', db_index=True, on_delete='CASCADE', null=False)), migrations.RenameField(model_name='projectbookmark', old_name='user', new_name='user_id'), migrations.AlterUniqueTogether(name='projectbookmark', unique_together={('project', 'user_id')})]
    return database_operations + [migrations.SeparateDatabaseAndState(state_operations=state_operations)]

def promptsactivity_migrations():
    if False:
        while True:
            i = 10
    database_operations = [migrations.AlterField(model_name='promptsactivity', name='user', field=sentry.db.models.fields.foreignkey.FlexibleForeignKey(to='sentry.User', db_constraint=False, db_index=True, null=False))]
    state_operations = [migrations.AlterField(model_name='promptsactivity', name='user', field=sentry.db.models.fields.hybrid_cloud_foreign_key.HybridCloudForeignKey('sentry.User', db_index=True, on_delete='CASCADE', null=False)), migrations.RenameField(model_name='promptsactivity', old_name='user', new_name='user_id')]
    return database_operations + [migrations.SeparateDatabaseAndState(state_operations=state_operations)]

def ruleactivity_migrations():
    if False:
        print('Hello World!')
    database_operations = [migrations.AlterField(model_name='ruleactivity', name='user', field=sentry.db.models.fields.foreignkey.FlexibleForeignKey(to='sentry.User', db_constraint=False, db_index=True, null=True))]
    state_operations = [migrations.AlterField(model_name='ruleactivity', name='user', field=sentry.db.models.fields.hybrid_cloud_foreign_key.HybridCloudForeignKey('sentry.User', db_index=True, on_delete='SET_NULL', null=True)), migrations.RenameField(model_name='ruleactivity', old_name='user', new_name='user_id')]
    return database_operations + [migrations.SeparateDatabaseAndState(state_operations=state_operations)]

class Migration(CheckedMigration):
    is_dangerous = False
    dependencies = [('sentry', '0396_add_usecase_to_indexer')]
    operations = apikey_migrations() + authprovider_migrations() + sentryapp_migrations() + sentryappinstallation_migrations() + sentryappinstallationforprovider_migrations() + organizationaccessrequest_migrations() + organizationonboardingtask_migrations() + projectbookmark_migrations() + promptsactivity_migrations() + ruleactivity_migrations() + organizationintegration_migrations()