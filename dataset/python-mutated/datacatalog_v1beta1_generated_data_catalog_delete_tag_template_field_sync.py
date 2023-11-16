from google.cloud import datacatalog_v1beta1

def sample_delete_tag_template_field():
    if False:
        print('Hello World!')
    client = datacatalog_v1beta1.DataCatalogClient()
    request = datacatalog_v1beta1.DeleteTagTemplateFieldRequest(name='name_value', force=True)
    client.delete_tag_template_field(request=request)