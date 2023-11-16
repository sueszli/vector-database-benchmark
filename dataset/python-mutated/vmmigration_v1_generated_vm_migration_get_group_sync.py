from google.cloud import vmmigration_v1

def sample_get_group():
    if False:
        i = 10
        return i + 15
    client = vmmigration_v1.VmMigrationClient()
    request = vmmigration_v1.GetGroupRequest(name='name_value')
    response = client.get_group(request=request)
    print(response)