from google.cloud import migrationcenter_v1

def sample_delete_group():
    if False:
        i = 10
        return i + 15
    client = migrationcenter_v1.MigrationCenterClient()
    request = migrationcenter_v1.DeleteGroupRequest(name='name_value')
    operation = client.delete_group(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)