from google.cloud import migrationcenter_v1

def sample_batch_delete_assets():
    if False:
        for i in range(10):
            print('nop')
    client = migrationcenter_v1.MigrationCenterClient()
    request = migrationcenter_v1.BatchDeleteAssetsRequest(parent='parent_value', names=['names_value1', 'names_value2'])
    client.batch_delete_assets(request=request)