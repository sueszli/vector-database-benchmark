from google.cloud import vmmigration_v1

def sample_update_target_project():
    if False:
        while True:
            i = 10
    client = vmmigration_v1.VmMigrationClient()
    request = vmmigration_v1.UpdateTargetProjectRequest()
    operation = client.update_target_project(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)