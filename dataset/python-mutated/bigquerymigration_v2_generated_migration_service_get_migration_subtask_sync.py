from google.cloud import bigquery_migration_v2

def sample_get_migration_subtask():
    if False:
        for i in range(10):
            print('nop')
    client = bigquery_migration_v2.MigrationServiceClient()
    request = bigquery_migration_v2.GetMigrationSubtaskRequest(name='name_value')
    response = client.get_migration_subtask(request=request)
    print(response)