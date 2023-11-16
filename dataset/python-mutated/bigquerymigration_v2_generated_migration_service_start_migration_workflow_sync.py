from google.cloud import bigquery_migration_v2

def sample_start_migration_workflow():
    if False:
        return 10
    client = bigquery_migration_v2.MigrationServiceClient()
    request = bigquery_migration_v2.StartMigrationWorkflowRequest(name='name_value')
    client.start_migration_workflow(request=request)