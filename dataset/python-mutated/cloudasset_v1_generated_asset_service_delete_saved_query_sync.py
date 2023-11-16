from google.cloud import asset_v1

def sample_delete_saved_query():
    if False:
        i = 10
        return i + 15
    client = asset_v1.AssetServiceClient()
    request = asset_v1.DeleteSavedQueryRequest(name='name_value')
    client.delete_saved_query(request=request)