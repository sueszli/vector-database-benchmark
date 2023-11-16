from google.cloud import vmmigration_v1

def sample_get_source():
    if False:
        i = 10
        return i + 15
    client = vmmigration_v1.VmMigrationClient()
    request = vmmigration_v1.GetSourceRequest(name='name_value')
    response = client.get_source(request=request)
    print(response)