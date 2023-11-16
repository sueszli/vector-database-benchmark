from google.cloud import retail_v2beta

def sample_get_default_branch():
    if False:
        i = 10
        return i + 15
    client = retail_v2beta.CatalogServiceClient()
    request = retail_v2beta.GetDefaultBranchRequest()
    response = client.get_default_branch(request=request)
    print(response)