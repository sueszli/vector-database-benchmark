from google.cloud import datacatalog_v1beta1

def sample_create_entry():
    if False:
        for i in range(10):
            print('nop')
    client = datacatalog_v1beta1.DataCatalogClient()
    entry = datacatalog_v1beta1.Entry()
    entry.type_ = 'FILESET'
    entry.integrated_system = 'CLOUD_PUBSUB'
    entry.gcs_fileset_spec.file_patterns = ['file_patterns_value1', 'file_patterns_value2']
    request = datacatalog_v1beta1.CreateEntryRequest(parent='parent_value', entry_id='entry_id_value', entry=entry)
    response = client.create_entry(request=request)
    print(response)