from google.cloud import datacatalog_v1

def sample_import_taxonomies():
    if False:
        for i in range(10):
            print('nop')
    client = datacatalog_v1.PolicyTagManagerSerializationClient()
    inline_source = datacatalog_v1.InlineSource()
    inline_source.taxonomies.display_name = 'display_name_value'
    request = datacatalog_v1.ImportTaxonomiesRequest(inline_source=inline_source, parent='parent_value')
    response = client.import_taxonomies(request=request)
    print(response)