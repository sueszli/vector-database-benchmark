from google.cloud import retail_v2

def sample_get_default_branch():
    if False:
        print('Hello World!')
    client = retail_v2.CatalogServiceClient()
    request = retail_v2.GetDefaultBranchRequest()
    response = client.get_default_branch(request=request)
    print(response)