from google.cloud import bigquery_migration_v2

def sample_list_migration_subtasks():
    if False:
        return 10
    client = bigquery_migration_v2.MigrationServiceClient()
    request = bigquery_migration_v2.ListMigrationSubtasksRequest(parent='parent_value')
    page_result = client.list_migration_subtasks(request=request)
    for response in page_result:
        print(response)