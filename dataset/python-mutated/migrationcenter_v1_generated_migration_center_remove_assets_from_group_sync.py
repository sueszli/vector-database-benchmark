from google.cloud import migrationcenter_v1

def sample_remove_assets_from_group():
    if False:
        print('Hello World!')
    client = migrationcenter_v1.MigrationCenterClient()
    assets = migrationcenter_v1.AssetList()
    assets.asset_ids = ['asset_ids_value1', 'asset_ids_value2']
    request = migrationcenter_v1.RemoveAssetsFromGroupRequest(group='group_value', assets=assets)
    operation = client.remove_assets_from_group(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)