from google.cloud import migrationcenter_v1

def sample_get_group():
    if False:
        return 10
    client = migrationcenter_v1.MigrationCenterClient()
    request = migrationcenter_v1.GetGroupRequest(name='name_value')
    response = client.get_group(request=request)
    print(response)