from google.cloud import bigquery_migration_v2alpha

def sample_create_migration_workflow():
    if False:
        for i in range(10):
            print('nop')
    client = bigquery_migration_v2alpha.MigrationServiceClient()
    request = bigquery_migration_v2alpha.CreateMigrationWorkflowRequest(parent='parent_value')
    response = client.create_migration_workflow(request=request)
    print(response)