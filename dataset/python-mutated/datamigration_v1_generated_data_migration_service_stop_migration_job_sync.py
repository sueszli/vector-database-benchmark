from google.cloud import clouddms_v1

def sample_stop_migration_job():
    if False:
        for i in range(10):
            print('nop')
    client = clouddms_v1.DataMigrationServiceClient()
    request = clouddms_v1.StopMigrationJobRequest()
    operation = client.stop_migration_job(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)