from google.cloud import datacatalog_v1beta1

def sample_update_tag():
    if False:
        i = 10
        return i + 15
    client = datacatalog_v1beta1.DataCatalogClient()
    tag = datacatalog_v1beta1.Tag()
    tag.column = 'column_value'
    tag.template = 'template_value'
    request = datacatalog_v1beta1.UpdateTagRequest(tag=tag)
    response = client.update_tag(request=request)
    print(response)