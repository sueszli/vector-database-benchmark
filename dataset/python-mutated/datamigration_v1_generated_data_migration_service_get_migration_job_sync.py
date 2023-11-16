from google.cloud import clouddms_v1

def sample_get_migration_job():
    if False:
        i = 10
        return i + 15
    client = clouddms_v1.DataMigrationServiceClient()
    request = clouddms_v1.GetMigrationJobRequest(name='name_value')
    response = client.get_migration_job(request=request)
    print(response)