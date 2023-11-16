from google.cloud import retail_v2alpha

def sample_get_default_branch():
    if False:
        print('Hello World!')
    client = retail_v2alpha.CatalogServiceClient()
    request = retail_v2alpha.GetDefaultBranchRequest()
    response = client.get_default_branch(request=request)
    print(response)