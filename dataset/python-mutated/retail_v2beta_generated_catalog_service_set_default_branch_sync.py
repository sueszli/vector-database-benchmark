from google.cloud import retail_v2beta

def sample_set_default_branch():
    if False:
        for i in range(10):
            print('nop')
    client = retail_v2beta.CatalogServiceClient()
    request = retail_v2beta.SetDefaultBranchRequest()
    client.set_default_branch(request=request)