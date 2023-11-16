from google.cloud import retail_v2alpha

def sample_set_default_branch():
    if False:
        print('Hello World!')
    client = retail_v2alpha.CatalogServiceClient()
    request = retail_v2alpha.SetDefaultBranchRequest()
    client.set_default_branch(request=request)