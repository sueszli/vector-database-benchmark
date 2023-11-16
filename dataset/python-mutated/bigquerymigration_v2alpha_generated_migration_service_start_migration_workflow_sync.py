from google.cloud import bigquery_migration_v2alpha

def sample_start_migration_workflow():
    if False:
        i = 10
        return i + 15
    client = bigquery_migration_v2alpha.MigrationServiceClient()
    request = bigquery_migration_v2alpha.StartMigrationWorkflowRequest(name='name_value')
    client.start_migration_workflow(request=request)