from google.cloud import vmmigration_v1

def sample_cancel_clone_job():
    if False:
        while True:
            i = 10
    client = vmmigration_v1.VmMigrationClient()
    request = vmmigration_v1.CancelCloneJobRequest(name='name_value')
    operation = client.cancel_clone_job(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)