from google.cloud import datacatalog_v1

def sample_update_tag():
    if False:
        print('Hello World!')
    client = datacatalog_v1.DataCatalogClient()
    tag = datacatalog_v1.Tag()
    tag.column = 'column_value'
    tag.template = 'template_value'
    request = datacatalog_v1.UpdateTagRequest(tag=tag)
    response = client.update_tag(request=request)
    print(response)