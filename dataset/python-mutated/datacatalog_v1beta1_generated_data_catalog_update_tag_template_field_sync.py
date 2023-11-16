from google.cloud import datacatalog_v1beta1

def sample_update_tag_template_field():
    if False:
        for i in range(10):
            print('nop')
    client = datacatalog_v1beta1.DataCatalogClient()
    tag_template_field = datacatalog_v1beta1.TagTemplateField()
    tag_template_field.type_.primitive_type = 'TIMESTAMP'
    request = datacatalog_v1beta1.UpdateTagTemplateFieldRequest(name='name_value', tag_template_field=tag_template_field)
    response = client.update_tag_template_field(request=request)
    print(response)