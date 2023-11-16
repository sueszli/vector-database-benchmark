from google.cloud import migrationcenter_v1

def sample_update_asset():
    if False:
        for i in range(10):
            print('nop')
    client = migrationcenter_v1.MigrationCenterClient()
    request = migrationcenter_v1.UpdateAssetRequest()
    response = client.update_asset(request=request)
    print(response)