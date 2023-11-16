from google.cloud import datacatalog_v1beta1

def sample_update_taxonomy():
    if False:
        i = 10
        return i + 15
    client = datacatalog_v1beta1.PolicyTagManagerClient()
    request = datacatalog_v1beta1.UpdateTaxonomyRequest()
    response = client.update_taxonomy(request=request)
    print(response)