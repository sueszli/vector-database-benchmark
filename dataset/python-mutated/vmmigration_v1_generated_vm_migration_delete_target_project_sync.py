from google.cloud import vmmigration_v1

def sample_delete_target_project():
    if False:
        for i in range(10):
            print('nop')
    client = vmmigration_v1.VmMigrationClient()
    request = vmmigration_v1.DeleteTargetProjectRequest(name='name_value')
    operation = client.delete_target_project(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)