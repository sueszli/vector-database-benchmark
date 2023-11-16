from google.cloud import bigquery_migration_v2

def sample_delete_migration_workflow():
    if False:
        i = 10
        return i + 15
    client = bigquery_migration_v2.MigrationServiceClient()
    request = bigquery_migration_v2.DeleteMigrationWorkflowRequest(name='name_value')
    client.delete_migration_workflow(request=request)