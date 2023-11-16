from google.cloud import datacatalog_v1beta1

def sample_update_tag_template():
    if False:
        for i in range(10):
            print('nop')
    client = datacatalog_v1beta1.DataCatalogClient()
    request = datacatalog_v1beta1.UpdateTagTemplateRequest()
    response = client.update_tag_template(request=request)
    print(response)