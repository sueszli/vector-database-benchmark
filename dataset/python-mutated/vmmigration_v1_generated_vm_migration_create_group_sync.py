from google.cloud import vmmigration_v1

def sample_create_group():
    if False:
        i = 10
        return i + 15
    client = vmmigration_v1.VmMigrationClient()
    request = vmmigration_v1.CreateGroupRequest(parent='parent_value', group_id='group_id_value')
    operation = client.create_group(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)