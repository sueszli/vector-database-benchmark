from google.cloud import vmmigration_v1

def sample_remove_group_migration():
    if False:
        for i in range(10):
            print('nop')
    client = vmmigration_v1.VmMigrationClient()
    request = vmmigration_v1.RemoveGroupMigrationRequest(group='group_value')
    operation = client.remove_group_migration(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)