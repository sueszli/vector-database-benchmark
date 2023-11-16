from google.cloud import migrationcenter_v1

def sample_update_group():
    if False:
        i = 10
        return i + 15
    client = migrationcenter_v1.MigrationCenterClient()
    request = migrationcenter_v1.UpdateGroupRequest()
    operation = client.update_group(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)