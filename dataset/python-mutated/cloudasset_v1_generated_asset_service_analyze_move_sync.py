from google.cloud import asset_v1

def sample_analyze_move():
    if False:
        for i in range(10):
            print('nop')
    client = asset_v1.AssetServiceClient()
    request = asset_v1.AnalyzeMoveRequest(resource='resource_value', destination_parent='destination_parent_value')
    response = client.analyze_move(request=request)
    print(response)