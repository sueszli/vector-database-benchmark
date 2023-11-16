from google.cloud import migrationcenter_v1

def sample_get_source():
    if False:
        i = 10
        return i + 15
    client = migrationcenter_v1.MigrationCenterClient()
    request = migrationcenter_v1.GetSourceRequest(name='name_value')
    response = client.get_source(request=request)
    print(response)