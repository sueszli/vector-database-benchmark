from google.cloud import datacatalog_v1beta1

def sample_update_entry():
    if False:
        while True:
            i = 10
    client = datacatalog_v1beta1.DataCatalogClient()
    entry = datacatalog_v1beta1.Entry()
    entry.type_ = 'FILESET'
    entry.integrated_system = 'CLOUD_PUBSUB'
    entry.gcs_fileset_spec.file_patterns = ['file_patterns_value1', 'file_patterns_value2']
    request = datacatalog_v1beta1.UpdateEntryRequest(entry=entry)
    response = client.update_entry(request=request)
    print(response)