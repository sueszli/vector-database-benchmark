from google.cloud import clouddms_v1

def sample_start_migration_job():
    if False:
        return 10
    client = clouddms_v1.DataMigrationServiceClient()
    request = clouddms_v1.StartMigrationJobRequest()
    operation = client.start_migration_job(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)