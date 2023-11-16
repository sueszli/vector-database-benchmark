from google.cloud import asset_v1

def sample_search_all_iam_policies():
    if False:
        print('Hello World!')
    client = asset_v1.AssetServiceClient()
    request = asset_v1.SearchAllIamPoliciesRequest(scope='scope_value')
    page_result = client.search_all_iam_policies(request=request)
    for response in page_result:
        print(response)