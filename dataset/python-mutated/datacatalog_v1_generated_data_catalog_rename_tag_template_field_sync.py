from google.cloud import datacatalog_v1

def sample_rename_tag_template_field():
    if False:
        print('Hello World!')
    client = datacatalog_v1.DataCatalogClient()
    request = datacatalog_v1.RenameTagTemplateFieldRequest(name='name_value', new_tag_template_field_id='new_tag_template_field_id_value')
    response = client.rename_tag_template_field(request=request)
    print(response)