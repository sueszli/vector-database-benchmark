from google.cloud import datacatalog_v1beta1

def sample_get_tag_template():
    if False:
        while True:
            i = 10
    client = datacatalog_v1beta1.DataCatalogClient()
    request = datacatalog_v1beta1.GetTagTemplateRequest(name='name_value')
    response = client.get_tag_template(request=request)
    print(response)