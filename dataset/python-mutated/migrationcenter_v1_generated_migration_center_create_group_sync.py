from google.cloud import migrationcenter_v1

def sample_create_group():
    if False:
        print('Hello World!')
    client = migrationcenter_v1.MigrationCenterClient()
    request = migrationcenter_v1.CreateGroupRequest(parent='parent_value', group_id='group_id_value')
    operation = client.create_group(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)