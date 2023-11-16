from google.cloud import migrationcenter_v1

def sample_delete_asset():
    if False:
        return 10
    client = migrationcenter_v1.MigrationCenterClient()
    request = migrationcenter_v1.DeleteAssetRequest(name='name_value')
    client.delete_asset(request=request)