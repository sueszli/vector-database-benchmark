from google.cloud import datacatalog_v1

def sample_rename_tag_template_field_enum_value():
    if False:
        while True:
            i = 10
    client = datacatalog_v1.DataCatalogClient()
    request = datacatalog_v1.RenameTagTemplateFieldEnumValueRequest(name='name_value', new_enum_value_display_name='new_enum_value_display_name_value')
    response = client.rename_tag_template_field_enum_value(request=request)
    print(response)