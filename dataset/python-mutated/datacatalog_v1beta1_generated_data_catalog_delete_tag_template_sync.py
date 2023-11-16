from google.cloud import datacatalog_v1beta1

def sample_delete_tag_template():
    if False:
        while True:
            i = 10
    client = datacatalog_v1beta1.DataCatalogClient()
    request = datacatalog_v1beta1.DeleteTagTemplateRequest(name='name_value', force=True)
    client.delete_tag_template(request=request)