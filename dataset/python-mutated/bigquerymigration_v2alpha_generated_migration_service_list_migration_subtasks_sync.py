from google.cloud import bigquery_migration_v2alpha

def sample_list_migration_subtasks():
    if False:
        for i in range(10):
            print('nop')
    client = bigquery_migration_v2alpha.MigrationServiceClient()
    request = bigquery_migration_v2alpha.ListMigrationSubtasksRequest(parent='parent_value')
    page_result = client.list_migration_subtasks(request=request)
    for response in page_result:
        print(response)