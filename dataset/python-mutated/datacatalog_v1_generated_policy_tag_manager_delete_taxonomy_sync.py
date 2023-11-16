from google.cloud import datacatalog_v1

def sample_delete_taxonomy():
    if False:
        i = 10
        return i + 15
    client = datacatalog_v1.PolicyTagManagerClient()
    request = datacatalog_v1.DeleteTaxonomyRequest(name='name_value')
    client.delete_taxonomy(request=request)