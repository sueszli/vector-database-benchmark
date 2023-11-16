from google.cloud import datacatalog_v1

def sample_get_tag_template():
    if False:
        for i in range(10):
            print('nop')
    client = datacatalog_v1.DataCatalogClient()
    request = datacatalog_v1.GetTagTemplateRequest(name='name_value')
    response = client.get_tag_template(request=request)
    print(response)