from google.cloud import retail_v2

def sample_set_default_branch():
    if False:
        i = 10
        return i + 15
    client = retail_v2.CatalogServiceClient()
    request = retail_v2.SetDefaultBranchRequest()
    client.set_default_branch(request=request)