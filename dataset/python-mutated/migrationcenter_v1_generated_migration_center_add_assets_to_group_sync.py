from google.cloud import migrationcenter_v1

def sample_add_assets_to_group():
    if False:
        return 10
    client = migrationcenter_v1.MigrationCenterClient()
    assets = migrationcenter_v1.AssetList()
    assets.asset_ids = ['asset_ids_value1', 'asset_ids_value2']
    request = migrationcenter_v1.AddAssetsToGroupRequest(group='group_value', assets=assets)
    operation = client.add_assets_to_group(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)