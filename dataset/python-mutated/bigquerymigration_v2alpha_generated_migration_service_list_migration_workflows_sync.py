from google.cloud import bigquery_migration_v2alpha

def sample_list_migration_workflows():
    if False:
        return 10
    client = bigquery_migration_v2alpha.MigrationServiceClient()
    request = bigquery_migration_v2alpha.ListMigrationWorkflowsRequest(parent='parent_value')
    page_result = client.list_migration_workflows(request=request)
    for response in page_result:
        print(response)