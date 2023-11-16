from google.cloud import datacatalog_v1

def sample_delete_tag_template_field():
    if False:
        i = 10
        return i + 15
    client = datacatalog_v1.DataCatalogClient()
    request = datacatalog_v1.DeleteTagTemplateFieldRequest(name='name_value', force=True)
    client.delete_tag_template_field(request=request)