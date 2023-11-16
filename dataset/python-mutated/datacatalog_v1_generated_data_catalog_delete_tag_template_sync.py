from google.cloud import datacatalog_v1

def sample_delete_tag_template():
    if False:
        return 10
    client = datacatalog_v1.DataCatalogClient()
    request = datacatalog_v1.DeleteTagTemplateRequest(name='name_value', force=True)
    client.delete_tag_template(request=request)