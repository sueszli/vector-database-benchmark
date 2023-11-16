from google.cloud import vmmigration_v1

def sample_update_source():
    if False:
        while True:
            i = 10
    client = vmmigration_v1.VmMigrationClient()
    request = vmmigration_v1.UpdateSourceRequest()
    operation = client.update_source(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)