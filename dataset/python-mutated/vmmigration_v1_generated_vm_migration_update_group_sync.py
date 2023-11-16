from google.cloud import vmmigration_v1

def sample_update_group():
    if False:
        print('Hello World!')
    client = vmmigration_v1.VmMigrationClient()
    request = vmmigration_v1.UpdateGroupRequest()
    operation = client.update_group(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)