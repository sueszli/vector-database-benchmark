from google.cloud import datacatalog_v1

def sample_get_taxonomy():
    if False:
        for i in range(10):
            print('nop')
    client = datacatalog_v1.PolicyTagManagerClient()
    request = datacatalog_v1.GetTaxonomyRequest(name='name_value')
    response = client.get_taxonomy(request=request)
    print(response)