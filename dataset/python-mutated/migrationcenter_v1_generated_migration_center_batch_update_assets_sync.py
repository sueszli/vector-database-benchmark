from google.cloud import migrationcenter_v1

def sample_batch_update_assets():
    if False:
        return 10
    client = migrationcenter_v1.MigrationCenterClient()
    request = migrationcenter_v1.BatchUpdateAssetsRequest(parent='parent_value')
    response = client.batch_update_assets(request=request)
    print(response)