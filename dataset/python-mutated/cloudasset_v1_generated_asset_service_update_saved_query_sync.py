from google.cloud import asset_v1

def sample_update_saved_query():
    if False:
        i = 10
        return i + 15
    client = asset_v1.AssetServiceClient()
    request = asset_v1.UpdateSavedQueryRequest()
    response = client.update_saved_query(request=request)
    print(response)