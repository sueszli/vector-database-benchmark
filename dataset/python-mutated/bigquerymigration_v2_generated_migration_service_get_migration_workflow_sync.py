from google.cloud import bigquery_migration_v2

def sample_get_migration_workflow():
    if False:
        for i in range(10):
            print('nop')
    client = bigquery_migration_v2.MigrationServiceClient()
    request = bigquery_migration_v2.GetMigrationWorkflowRequest(name='name_value')
    response = client.get_migration_workflow(request=request)
    print(response)