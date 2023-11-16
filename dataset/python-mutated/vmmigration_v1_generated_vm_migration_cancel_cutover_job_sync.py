from google.cloud import vmmigration_v1

def sample_cancel_cutover_job():
    if False:
        print('Hello World!')
    client = vmmigration_v1.VmMigrationClient()
    request = vmmigration_v1.CancelCutoverJobRequest(name='name_value')
    operation = client.cancel_cutover_job(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)