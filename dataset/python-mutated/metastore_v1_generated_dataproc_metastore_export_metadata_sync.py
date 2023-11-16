from google.cloud import metastore_v1

def sample_export_metadata():
    if False:
        print('Hello World!')
    client = metastore_v1.DataprocMetastoreClient()
    request = metastore_v1.ExportMetadataRequest(destination_gcs_folder='destination_gcs_folder_value', service='service_value')
    operation = client.export_metadata(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)