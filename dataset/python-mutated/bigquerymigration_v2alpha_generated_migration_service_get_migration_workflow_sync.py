from google.cloud import bigquery_migration_v2alpha

def sample_get_migration_workflow():
    if False:
        return 10
    client = bigquery_migration_v2alpha.MigrationServiceClient()
    request = bigquery_migration_v2alpha.GetMigrationWorkflowRequest(name='name_value')
    response = client.get_migration_workflow(request=request)
    print(response)