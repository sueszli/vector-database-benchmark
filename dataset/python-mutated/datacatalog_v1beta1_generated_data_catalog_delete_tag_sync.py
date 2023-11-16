from google.cloud import datacatalog_v1beta1

def sample_delete_tag():
    if False:
        print('Hello World!')
    client = datacatalog_v1beta1.DataCatalogClient()
    request = datacatalog_v1beta1.DeleteTagRequest(name='name_value')
    client.delete_tag(request=request)