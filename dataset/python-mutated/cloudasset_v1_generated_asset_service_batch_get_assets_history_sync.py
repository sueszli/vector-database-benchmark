from google.cloud import asset_v1

def sample_batch_get_assets_history():
    if False:
        while True:
            i = 10
    client = asset_v1.AssetServiceClient()
    request = asset_v1.BatchGetAssetsHistoryRequest(parent='parent_value')
    response = client.batch_get_assets_history(request=request)
    print(response)