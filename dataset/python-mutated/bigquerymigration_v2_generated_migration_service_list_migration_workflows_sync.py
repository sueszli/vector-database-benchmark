from google.cloud import bigquery_migration_v2

def sample_list_migration_workflows():
    if False:
        while True:
            i = 10
    client = bigquery_migration_v2.MigrationServiceClient()
    request = bigquery_migration_v2.ListMigrationWorkflowsRequest(parent='parent_value')
    page_result = client.list_migration_workflows(request=request)
    for response in page_result:
        print(response)