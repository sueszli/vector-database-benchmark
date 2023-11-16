from google.cloud import datacatalog_v1

def sample_create_taxonomy():
    if False:
        i = 10
        return i + 15
    client = datacatalog_v1.PolicyTagManagerClient()
    request = datacatalog_v1.CreateTaxonomyRequest(parent='parent_value')
    response = client.create_taxonomy(request=request)
    print(response)