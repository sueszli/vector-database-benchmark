from google.cloud import bigquery_migration_v2alpha

def sample_get_migration_subtask():
    if False:
        print('Hello World!')
    client = bigquery_migration_v2alpha.MigrationServiceClient()
    request = bigquery_migration_v2alpha.GetMigrationSubtaskRequest(name='name_value')
    response = client.get_migration_subtask(request=request)
    print(response)