from google.cloud import vmmigration_v1

def sample_delete_group():
    if False:
        i = 10
        return i + 15
    client = vmmigration_v1.VmMigrationClient()
    request = vmmigration_v1.DeleteGroupRequest(name='name_value')
    operation = client.delete_group(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)