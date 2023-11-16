from google.cloud import datacatalog_v1

def sample_create_tag():
    if False:
        while True:
            i = 10
    client = datacatalog_v1.DataCatalogClient()
    tag = datacatalog_v1.Tag()
    tag.column = 'column_value'
    tag.template = 'template_value'
    request = datacatalog_v1.CreateTagRequest(parent='parent_value', tag=tag)
    response = client.create_tag(request=request)
    print(response)