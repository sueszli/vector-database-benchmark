from google.cloud import datacatalog_v1beta1

def sample_create_taxonomy():
    if False:
        return 10
    client = datacatalog_v1beta1.PolicyTagManagerClient()
    request = datacatalog_v1beta1.CreateTaxonomyRequest(parent='parent_value')
    response = client.create_taxonomy(request=request)
    print(response)