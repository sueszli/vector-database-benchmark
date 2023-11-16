from google.cloud import clouddms_v1

def sample_restart_migration_job():
    if False:
        print('Hello World!')
    client = clouddms_v1.DataMigrationServiceClient()
    request = clouddms_v1.RestartMigrationJobRequest()
    operation = client.restart_migration_job(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)