from google.cloud import vmmigration_v1

def sample_delete_source():
    if False:
        for i in range(10):
            print('nop')
    client = vmmigration_v1.VmMigrationClient()
    request = vmmigration_v1.DeleteSourceRequest(name='name_value')
    operation = client.delete_source(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)