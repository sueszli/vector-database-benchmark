from google.cloud import metastore_v1beta

def sample_export_metadata():
    if False:
        i = 10
        return i + 15
    client = metastore_v1beta.DataprocMetastoreClient()
    request = metastore_v1beta.ExportMetadataRequest(destination_gcs_folder='destination_gcs_folder_value', service='service_value')
    operation = client.export_metadata(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)