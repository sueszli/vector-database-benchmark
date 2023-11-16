from google.cloud import datacatalog_v1beta1

def sample_get_taxonomy():
    if False:
        while True:
            i = 10
    client = datacatalog_v1beta1.PolicyTagManagerClient()
    request = datacatalog_v1beta1.GetTaxonomyRequest(name='name_value')
    response = client.get_taxonomy(request=request)
    print(response)