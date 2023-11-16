from google.cloud import datacatalog_v1

def sample_create_entry():
    if False:
        while True:
            i = 10
    client = datacatalog_v1.DataCatalogClient()
    entry = datacatalog_v1.Entry()
    entry.type_ = 'LOOK'
    entry.integrated_system = 'VERTEX_AI'
    entry.gcs_fileset_spec.file_patterns = ['file_patterns_value1', 'file_patterns_value2']
    request = datacatalog_v1.CreateEntryRequest(parent='parent_value', entry_id='entry_id_value', entry=entry)
    response = client.create_entry(request=request)
    print(response)