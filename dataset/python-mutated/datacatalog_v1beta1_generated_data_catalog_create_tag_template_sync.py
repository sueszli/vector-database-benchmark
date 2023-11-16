from google.cloud import datacatalog_v1beta1

def sample_create_tag_template():
    if False:
        while True:
            i = 10
    client = datacatalog_v1beta1.DataCatalogClient()
    request = datacatalog_v1beta1.CreateTagTemplateRequest(parent='parent_value', tag_template_id='tag_template_id_value')
    response = client.create_tag_template(request=request)
    print(response)