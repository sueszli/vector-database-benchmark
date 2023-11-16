from google.cloud import vmmigration_v1

def sample_create_target_project():
    if False:
        return 10
    client = vmmigration_v1.VmMigrationClient()
    request = vmmigration_v1.CreateTargetProjectRequest(parent='parent_value', target_project_id='target_project_id_value')
    operation = client.create_target_project(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)