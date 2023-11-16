from google.cloud import datacatalog_v1

def sample_update_tag_template():
    if False:
        i = 10
        return i + 15
    client = datacatalog_v1.DataCatalogClient()
    request = datacatalog_v1.UpdateTagTemplateRequest()
    response = client.update_tag_template(request=request)
    print(response)