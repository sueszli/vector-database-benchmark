from google.cloud import datacatalog_v1beta1

def sample_export_taxonomies():
    if False:
        return 10
    client = datacatalog_v1beta1.PolicyTagManagerSerializationClient()
    request = datacatalog_v1beta1.ExportTaxonomiesRequest(serialized_taxonomies=True, parent='parent_value', taxonomies=['taxonomies_value1', 'taxonomies_value2'])
    response = client.export_taxonomies(request=request)
    print(response)