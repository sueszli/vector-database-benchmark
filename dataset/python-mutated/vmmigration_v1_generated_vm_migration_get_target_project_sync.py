from google.cloud import vmmigration_v1

def sample_get_target_project():
    if False:
        return 10
    client = vmmigration_v1.VmMigrationClient()
    request = vmmigration_v1.GetTargetProjectRequest(name='name_value')
    response = client.get_target_project(request=request)
    print(response)