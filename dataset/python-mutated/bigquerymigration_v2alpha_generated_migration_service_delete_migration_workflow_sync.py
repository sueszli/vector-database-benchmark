from google.cloud import bigquery_migration_v2alpha

def sample_delete_migration_workflow():
    if False:
        return 10
    client = bigquery_migration_v2alpha.MigrationServiceClient()
    request = bigquery_migration_v2alpha.DeleteMigrationWorkflowRequest(name='name_value')
    client.delete_migration_workflow(request=request)