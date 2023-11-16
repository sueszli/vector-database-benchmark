from google.cloud import datacatalog_v1beta1

def sample_delete_taxonomy():
    if False:
        print('Hello World!')
    client = datacatalog_v1beta1.PolicyTagManagerClient()
    request = datacatalog_v1beta1.DeleteTaxonomyRequest(name='name_value')
    client.delete_taxonomy(request=request)