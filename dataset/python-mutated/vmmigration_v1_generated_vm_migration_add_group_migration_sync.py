from google.cloud import vmmigration_v1

def sample_add_group_migration():
    if False:
        for i in range(10):
            print('nop')
    client = vmmigration_v1.VmMigrationClient()
    request = vmmigration_v1.AddGroupMigrationRequest(group='group_value')
    operation = client.add_group_migration(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)