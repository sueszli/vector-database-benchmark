from google.cloud import datacatalog_v1beta1

def sample_create_tag_template_field():
    if False:
        i = 10
        return i + 15
    client = datacatalog_v1beta1.DataCatalogClient()
    tag_template_field = datacatalog_v1beta1.TagTemplateField()
    tag_template_field.type_.primitive_type = 'TIMESTAMP'
    request = datacatalog_v1beta1.CreateTagTemplateFieldRequest(parent='parent_value', tag_template_field_id='tag_template_field_id_value', tag_template_field=tag_template_field)
    response = client.create_tag_template_field(request=request)
    print(response)