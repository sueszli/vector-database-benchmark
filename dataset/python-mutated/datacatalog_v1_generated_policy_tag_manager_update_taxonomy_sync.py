from google.cloud import datacatalog_v1

def sample_update_taxonomy():
    if False:
        print('Hello World!')
    client = datacatalog_v1.PolicyTagManagerClient()
    request = datacatalog_v1.UpdateTaxonomyRequest()
    response = client.update_taxonomy(request=request)
    print(response)