from google.cloud import datacatalog_v1

def sample_delete_tag():
    if False:
        return 10
    client = datacatalog_v1.DataCatalogClient()
    request = datacatalog_v1.DeleteTagRequest(name='name_value')
    client.delete_tag(request=request)