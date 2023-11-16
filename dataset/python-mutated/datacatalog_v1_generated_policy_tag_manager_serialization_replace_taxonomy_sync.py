from google.cloud import datacatalog_v1

def sample_replace_taxonomy():
    if False:
        i = 10
        return i + 15
    client = datacatalog_v1.PolicyTagManagerSerializationClient()
    serialized_taxonomy = datacatalog_v1.SerializedTaxonomy()
    serialized_taxonomy.display_name = 'display_name_value'
    request = datacatalog_v1.ReplaceTaxonomyRequest(name='name_value', serialized_taxonomy=serialized_taxonomy)
    response = client.replace_taxonomy(request=request)
    print(response)