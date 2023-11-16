from google.cloud import datacatalog_v1beta1

def sample_create_tag():
    if False:
        print('Hello World!')
    client = datacatalog_v1beta1.DataCatalogClient()
    tag = datacatalog_v1beta1.Tag()
    tag.column = 'column_value'
    tag.template = 'template_value'
    request = datacatalog_v1beta1.CreateTagRequest(parent='parent_value', tag=tag)
    response = client.create_tag(request=request)
    print(response)