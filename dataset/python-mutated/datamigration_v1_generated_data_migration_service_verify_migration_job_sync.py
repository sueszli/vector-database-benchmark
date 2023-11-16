from google.cloud import clouddms_v1

def sample_verify_migration_job():
    if False:
        while True:
            i = 10
    client = clouddms_v1.DataMigrationServiceClient()
    request = clouddms_v1.VerifyMigrationJobRequest()
    operation = client.verify_migration_job(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)