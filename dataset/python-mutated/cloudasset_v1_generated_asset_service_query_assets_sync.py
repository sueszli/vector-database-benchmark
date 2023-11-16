from google.cloud import asset_v1

def sample_query_assets():
    if False:
        for i in range(10):
            print('nop')
    client = asset_v1.AssetServiceClient()
    request = asset_v1.QueryAssetsRequest(statement='statement_value', parent='parent_value')
    response = client.query_assets(request=request)
    print(response)