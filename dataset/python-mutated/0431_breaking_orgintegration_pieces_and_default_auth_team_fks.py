from django.db import migrations
import sentry.db.models.fields.bounded
import sentry.db.models.fields.hybrid_cloud_foreign_key
from sentry.new_migrations.migrations import CheckedMigration

def authprovider_migrations():
    if False:
        print('Hello World!')
    return [migrations.SeparateDatabaseAndState(state_operations=[migrations.RemoveField(model_name='authprovider', name='default_teams')])]

def pagerdutyservice_migrations():
    if False:
        return 10
    database_operations = [migrations.AlterField(model_name='pagerdutyservice', name='organization_integration', field=sentry.db.models.fields.foreignkey.FlexibleForeignKey(to='sentry.OrganizationIntegration', db_constraint=False, db_index=True, null=False))]
    state_operations = [migrations.AlterField(model_name='pagerdutyservice', name='organization_integration', field=sentry.db.models.fields.hybrid_cloud_foreign_key.HybridCloudForeignKey('sentry.OrganizationIntegration', db_index=True, on_delete='CASCADE', null=False)), migrations.RenameField(model_name='pagerdutyservice', old_name='organization_integration', new_name='organization_integration_id')]
    return database_operations + [migrations.SeparateDatabaseAndState(state_operations=state_operations)]

def repositoryprojectpathconfig_migrations():
    if False:
        i = 10
        return i + 15
    database_operations = [migrations.AlterField(model_name='repositoryprojectpathconfig', name='organization_integration', field=sentry.db.models.fields.foreignkey.FlexibleForeignKey(to='sentry.OrganizationIntegration', db_constraint=False, db_index=True, null=False))]
    state_operations = [migrations.AlterField(model_name='repositoryprojectpathconfig', name='organization_integration', field=sentry.db.models.fields.hybrid_cloud_foreign_key.HybridCloudForeignKey('sentry.OrganizationIntegration', db_index=True, on_delete='CASCADE', null=False)), migrations.RenameField(model_name='repositoryprojectpathconfig', old_name='organization_integration', new_name='organization_integration_id')]
    return database_operations + [migrations.SeparateDatabaseAndState(state_operations=state_operations)]

def authproviderdefaultteams_migrations():
    if False:
        return 10
    database_operations = [migrations.AlterField(model_name='authproviderdefaultteams', name='authprovider', field=sentry.db.models.fields.foreignkey.FlexibleForeignKey(to='sentry.OrganizationIntegration', db_constraint=False, db_index=False, null=False)), migrations.AlterField(model_name='authproviderdefaultteams', name='team', field=sentry.db.models.fields.foreignkey.FlexibleForeignKey(to='sentry.OrganizationIntegration', db_constraint=False, db_index=False, null=False)), migrations.AlterUniqueTogether(name='authproviderdefaultteams', unique_together=set())]
    state_operations = [migrations.AlterField(model_name='authproviderdefaultteams', name='authprovider', field=sentry.db.models.fields.BoundedBigIntegerField()), migrations.AlterField(model_name='authproviderdefaultteams', name='team', field=sentry.db.models.fields.BoundedBigIntegerField()), migrations.RenameField(model_name='authproviderdefaultteams', old_name='authprovider', new_name='authprovider_id'), migrations.RenameField(model_name='authproviderdefaultteams', old_name='team', new_name='team_id')]
    return database_operations + [migrations.SeparateDatabaseAndState(state_operations=state_operations)]

class Migration(CheckedMigration):
    is_dangerous = False
    dependencies = [('sentry', '0430_break_fks_on_alertruletriggeraction')]
    operations = authprovider_migrations() + pagerdutyservice_migrations() + repositoryprojectpathconfig_migrations() + authproviderdefaultteams_migrations()