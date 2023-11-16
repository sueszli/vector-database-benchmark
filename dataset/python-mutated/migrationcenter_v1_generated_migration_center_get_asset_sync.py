from google.cloud import migrationcenter_v1

def sample_get_asset():
    if False:
        for i in range(10):
            print('nop')
    client = migrationcenter_v1.MigrationCenterClient()
    request = migrationcenter_v1.GetAssetRequest(name='name_value')
    response = client.get_asset(request=request)
    print(response)