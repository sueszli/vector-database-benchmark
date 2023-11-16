from google.cloud import documentai_v1beta3

def sample_update_dataset_schema():
    if False:
        while True:
            i = 10
    client = documentai_v1beta3.DocumentServiceClient()
    request = documentai_v1beta3.UpdateDatasetSchemaRequest()
    response = client.update_dataset_schema(request=request)
    print(response)