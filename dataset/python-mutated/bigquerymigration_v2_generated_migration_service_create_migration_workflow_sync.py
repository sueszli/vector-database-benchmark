from google.cloud import bigquery_migration_v2

def sample_create_migration_workflow():
    if False:
        while True:
            i = 10
    client = bigquery_migration_v2.MigrationServiceClient()
    request = bigquery_migration_v2.CreateMigrationWorkflowRequest(parent='parent_value')
    response = client.create_migration_workflow(request=request)
    print(response)