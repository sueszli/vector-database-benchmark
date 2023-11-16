from google.cloud import documentai_v1beta3

def sample_get_dataset_schema():
    if False:
        i = 10
        return i + 15
    client = documentai_v1beta3.DocumentServiceClient()
    request = documentai_v1beta3.GetDatasetSchemaRequest(name='name_value')
    response = client.get_dataset_schema(request=request)
    print(response)